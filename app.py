#!/usr/bin/env python3
# ==============================================================================
#
# Gemini-CLI Telegram Bot (Python Version)
#
# Description:
# This script acts as a long-running bot that fetches messages from Telegram,
# processes them as prompts for the Gemini CLI, and sends the results back.
# It allows for remote interaction with the Gemini agent, including switching
# between different project contexts.
#
# Usage:
# python3 telegram_bot.py
#
# Prerequisites:
# - Python 3.6+
# - requests library: pip install requests
# - python-dotenv library: pip install python-dotenv
# - gemini-cli: The Gemini command-line interface must be installed and
#   configured in the system's PATH.
# - An active internet connection.
#
# Setup:
# 1. Create a file named .env in the same directory.
# 2. Add your Telegram bot token to the .env file like this:
#    TELEGRAM_BOT_TOKEN="12345:your_actual_token_here"
# 3. Update the configuration variables in the "--- Configuration ---" section below.
# 4. Run the script: python3 telegram_bot.py
#
# ==============================================================================

import os
import sys
import json
import logging
import re
import shlex
import shutil
import subprocess
import time
import threading
from queue import Queue, Empty
from pathlib import Path

from dotenv import load_dotenv
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google.cloud import speech

# --- Configuration ---

load_dotenv()  # Load environment variables from .env file

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcloud_credentials.json"

# Enable or disable debug mode for Gemini CLI.
DEBUG_MODE = False

# Your Telegram User ID (loaded from .env file)
AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID")

# Your Telegram Bot Token (loaded from .env file)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# The base directory where all your projects are stored.
PROJECTS_DIR = Path(__file__).parent / "projects"

# Path to the Gemini settings file to be copied into new projects.
GEMINI_SETTINGS_FILE = Path.home() / ".gemini" / "settings.json"

# File to store chat ID to project path mappings and last update ID.
CONTEXT_FILE = "project_contexts.json"

# Log file for debugging.
LOG_FILE = "telegram-bot.log"

# Telegram API URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Streaming configuration
STREAM_MODE = os.getenv("STREAM_MODE", "partial").lower()  # partial | block | off
STREAM_UPDATE_INTERVAL = float(os.getenv("STREAM_UPDATE_INTERVAL", "1.5"))
STREAM_MIN_CHARS = int(os.getenv("STREAM_MIN_CHARS", "200"))
STREAM_MAX_CHARS = int(os.getenv("STREAM_MAX_CHARS", "800"))
STREAM_TAIL_LIMIT = int(os.getenv("STREAM_TAIL_LIMIT", "3800"))
STREAM_CURSOR = os.getenv("STREAM_CURSOR", " ▌")

# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- State Management ---

LAST_THREAD_ID = {}

def load_state():
    """Loads the bot state from the context file."""
    if not Path(CONTEXT_FILE).exists():
        logging.info(f"Context file not found. Creating a new one at: {CONTEXT_FILE}")
        return {"contexts": {}, "last_update_id": 0, "prompt_counters": {}, "context_workflows": {}, "awaiting_input": {}, "force_new_session": {}}
    try:
        with open(CONTEXT_FILE, 'r') as f:
            state = json.load(f)
            # Ensure all keys are present for backward compatibility
            state.setdefault("prompt_counters", {})
            state.setdefault("context_workflows", {})
            state.setdefault("awaiting_input", {})
            state.setdefault("force_new_session", {})
            return state
    except (json.JSONDecodeError, FileNotFoundError):
        logging.error(f"Could not read or parse {CONTEXT_FILE}. Starting fresh.")
        return {"contexts": {}, "last_update_id": 0, "prompt_counters": {}, "context_workflows": {}, "awaiting_input": {}}

def save_state(state):
    """Saves the bot state to the context file."""
    try:
        with open(CONTEXT_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    except IOError as e:
        logging.error(f"Could not write to state file {CONTEXT_FILE}: {e}")

# --- Telegram API Helpers ---

def format_for_telegram(text):
    """Formats text for Telegram's MarkdownV1, handling GFM features and escaping."""
    # Split by code blocks and inline code to avoid modifying them
    parts = re.split(r'(```[\s\S]*?```|`[^`]*?`)', text)

    for i in range(0, len(parts), 2):  # Process parts outside code blocks
        part = parts[i]

        # Convert GFM bold `**text**` to MarkdownV1 `*text*`
        part = re.sub(r'\*\*(.*?)\*\*', r'*\1*', part)

        # Escape underscores `_` to prevent them from being interpreted as italics
        part = part.replace('_', r'\_')

        # MarkdownV1 doesn't support lists. Let's convert `* ` to a bullet point.
        part = re.sub(r'^\s*\*\s+', '• ', part, flags=re.MULTILINE)

        # Headers to bold
        part = re.sub(r'^# (.*?)$', r'*\1*', part, flags=re.MULTILINE)
        part = re.sub(r'^## (.*?)$', r'*\1*', part, flags=re.MULTILINE)
        part = re.sub(r'^### (.*?)$', r'*\1*', part, flags=re.MULTILINE)

        parts[i] = part

    return "".join(parts)

def break_sentences_into_lines(text):
    """Adds a newline after each sentence, preserving code blocks."""
    if not text:
        return ""
    
    parts = re.split(r'(```[\s\S]*?```)', text)
    for i in range(0, len(parts), 2):
        part = parts[i]
        # Use a lookbehind to keep the punctuation, and replace the following space with a newline.
        part = re.sub(r'(?<=[.!?])\s+', '\n', part)
        parts[i] = part
        
    return "".join(parts)

def format_for_telegram_paragraphs(text):
    """
    Ensures text has proper paragraph spacing for Telegram Markdown,
    while preserving code blocks.
    """
    if not text:
        return ""
    
    # Process text outside of code blocks
    parts = re.split(r'(```[\s\S]*?```)', text)
    for i in range(0, len(parts), 2):
        part = parts[i]
        # Replace single newlines with double, then collapse excessive newlines.
        # This effectively creates paragraphs from single-newline-separated text.
        part = part.replace('\n', '\n\n')
        part = re.sub(r'\n{3,}', '\n\n', part)
        parts[i] = part
    
    return "".join(parts)

def send_message(chat_id, text, parse_mode="Markdown", message_thread_id=None):
    """Sends a text message to a Telegram chat."""
    logging.info(f"Sending message to Chat ID: {chat_id}")
    if not text:
        logging.warning("Attempted to send an empty message. Aborting.")
        return

    if parse_mode == "Markdown":
        text = format_for_telegram(text)

    if message_thread_id is None:
        message_thread_id = LAST_THREAD_ID.get(str(chat_id))

    payload = {
        'chat_id': chat_id,
        'text': text
    }
    if message_thread_id is not None:
        payload['message_thread_id'] = message_thread_id
    if parse_mode:
        payload['parse_mode'] = parse_mode
    try:
        response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        if response_json.get("ok"):
            logging.info(f"Successfully sent message to Chat ID: {chat_id}.")
        else:
            logging.error(f"Error in Telegram API response when sending message: {response_json}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to Chat ID: {chat_id}. Request failed: {e}")

def send_message_raw(chat_id, text, message_thread_id=None):
    """Sends a plain text message and returns message_id on success."""
    if not text:
        return None
    if message_thread_id is None:
        message_thread_id = LAST_THREAD_ID.get(str(chat_id))
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    if message_thread_id is not None:
        payload['message_thread_id'] = message_thread_id
    try:
        response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        if response_json.get("ok"):
            return response_json.get("result", {}).get("message_id")
        logging.error(f"Error in Telegram API response when sending raw message: {response_json}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending raw message to Chat ID: {chat_id}. Request failed: {e}")
    return None

def send_message_with_id(chat_id, text, parse_mode=None, message_thread_id=None):
    """Sends a message and returns message_id on success."""
    if not text:
        return None
    if parse_mode == "Markdown":
        text = format_for_telegram(text)
    if message_thread_id is None:
        message_thread_id = LAST_THREAD_ID.get(str(chat_id))
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    if message_thread_id is not None:
        payload['message_thread_id'] = message_thread_id
    if parse_mode:
        payload['parse_mode'] = parse_mode
    try:
        response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        if response_json.get("ok"):
            return response_json.get("result", {}).get("message_id")
        logging.error(f"Error in Telegram API response when sending message: {response_json}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to Chat ID: {chat_id}. Request failed: {e}")
    return None

def edit_message_text(chat_id, message_id, text, parse_mode=None):
    """Edits a Telegram message. Returns True on success."""
    if not text:
        return False
    payload = {
        'chat_id': chat_id,
        'message_id': message_id,
        'text': text
    }
    if parse_mode:
        payload['parse_mode'] = parse_mode

    try:
        response = requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload, timeout=30)
        if response.status_code == 429:
            try:
                retry_after = response.json().get("parameters", {}).get("retry_after", 1)
            except Exception:
                retry_after = 1
            time.sleep(min(retry_after, 5))
            return False
        response.raise_for_status()
        response_json = response.json()
        if response_json.get("ok"):
            return True
        description = response_json.get("description", "")
        if "message is not modified" in description.lower():
            return True
        logging.debug(f"Edit message failed: {response_json}")
        return False
    except requests.exceptions.RequestException as e:
        logging.debug(f"Edit message request failed: {e}")
        return False

def send_file(chat_id, file_path, message_thread_id=None):
    """Sends a file to a Telegram chat."""
    logging.info(f"Sending file to Chat ID: {chat_id}, File: {file_path}")
    file_path = Path(file_path)  # Ensure file_path is a Path object
    if not file_path.is_file():
        logging.error(f"File not found for sending: {file_path}")
        send_message(chat_id, f"Error: Could not find file `{file_path.name}` on the server.", message_thread_id=message_thread_id)
        return
        
    try:
        with open(file_path, 'rb') as f:
            files = {'document': f}
            payload = {'chat_id': chat_id}
            if message_thread_id is None:
                message_thread_id = LAST_THREAD_ID.get(str(chat_id))
            if message_thread_id is not None:
                payload['message_thread_id'] = message_thread_id
            response = requests.post(f"{TELEGRAM_API_URL}/sendDocument", data=payload, files=files, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("ok"):
                logging.info(f"Successfully sent file to Chat ID: {chat_id}.")
            else:
                logging.error(f"Error in Telegram API response when sending file: {response_json}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending file to Chat ID: {chat_id}. Request failed: {e}")
        send_message(chat_id, f"An error occurred while sending the file `{file_path.name}`.", message_thread_id=message_thread_id)

# --- Icon Helper ---

def get_file_icon(filename):
    """Returns an icon for a given filename based on its extension."""
    if filename.endswith('.py'):
        return "🐍"
    elif filename.endswith('.md'):
        return "⭐"
    elif filename.endswith('.log'):
        return "📜"
    elif filename.endswith('.txt'):
        return "📝"
    elif filename.endswith('.sh'):
        return "📜"
    elif filename.endswith(('.bat', '.cmd', '.exe')):
        return "🔴"
    elif filename.endswith('.json'):
        return "🧩"
    elif filename.endswith('.env'):
        return "🔑"
    else:
        return "📄"

# --- Command Handlers ---

def set_project(chat_id, project_name, state, initial_prompt=None, message_thread_id=None):
    """Sets the project context for a given chat and returns status and message."""
    project_path = Path(PROJECTS_DIR) / project_name
    logging.info(f"Attempting to set project path to: {project_path}")

    if project_path.is_dir():
        state["contexts"][str(chat_id)] = str(project_path)
        gemini_md_path = project_path / "GEMINI.md"
        if not gemini_md_path.exists():
            gemini_md_path.write_text("# Project Requirements\n\n", encoding='utf-8')
            logging.info(f"Created GEMINI.md for existing project at: {project_path}")
        
        start_file_observer(chat_id, str(project_path))

        if initial_prompt:
            logging.info(f"Handling initial prompt for selected project: {initial_prompt}")
            handle_gemini_prompt(chat_id, initial_prompt, state, message_thread_id=message_thread_id)
        
        message_text = f"Project context set to: `{project_path}`"
        return True, message_text
    else:
        message_text = f"Error: Project `{project_name}` not found in `{PROJECTS_DIR}`."
        return False, message_text

def handle_set_project(chat_id, text, state, message_thread_id=None):
    """Handles the /set_project command."""
    parts = text.split()
    if len(parts) < 2:
        # List projects as buttons
        try:
            projects = [d for d in os.listdir(PROJECTS_DIR) if (PROJECTS_DIR / d).is_dir()]
            if not projects:
                send_message(chat_id, "No projects found.")
                return

            keyboard = {
                "inline_keyboard": [[{"text": f"📂 {p}", "callback_data": f"set_project:{p}"}] for p in projects]
            }
            keyboard["inline_keyboard"].append([{"text": "➕ New Project", "callback_data": "new_project_prompt"}])
            payload = {
                'chat_id': chat_id,
                'text': "Select a project:",
                'reply_markup': json.dumps(keyboard)
            }
            thread_id = LAST_THREAD_ID.get(str(chat_id))
            if thread_id is not None:
                payload['message_thread_id'] = thread_id
            response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Error creating project list: {e}")
            send_message(chat_id, "An error occurred while listing projects.")
        return

    project_name = parts[1]
    initial_prompt = " ".join(parts[2:])
    success, message = set_project(chat_id, project_name, state, initial_prompt, message_thread_id=message_thread_id)
    send_message(chat_id, message, message_thread_id=message_thread_id)

def create_new_project(chat_id, project_name, state, initial_prompt=None, message_thread_id=None):
    """Creates a new project directory and sets it as the current context."""
    project_path = Path(PROJECTS_DIR) / project_name
    logging.info(f"Checking if project path exists: {project_path}")

    if project_path.exists():
        send_message(chat_id, f"Error: Project `{project_name}` already exists in `{PROJECTS_DIR}`.")
    else:
        try:
            project_path.mkdir(parents=True, exist_ok=True)
            (project_path / "GEMINI.md").write_text("# Project Requirements\n\n", encoding='utf-8')
            if Path(GEMINI_SETTINGS_FILE).is_file():
                shutil.copy(GEMINI_SETTINGS_FILE, project_path / "settings.json")
                logging.info(f"Copied Gemini settings to {project_path / 'settings.json'}")
            else:
                logging.warning(f"Gemini settings file not found at '{GEMINI_SETTINGS_FILE}'. Skipping copy.")
            state["contexts"][str(chat_id)] = str(project_path)
            send_message(chat_id, f"Project `{project_name}` created and context set to: `{project_path}`", message_thread_id=message_thread_id)
            start_file_observer(chat_id, str(project_path))

            if initial_prompt:
                logging.info(f"Handling initial prompt for new project: {initial_prompt}")
                handle_gemini_prompt(chat_id, initial_prompt, state, message_thread_id=message_thread_id)

        except OSError as e:
            logging.error(f"Failed to create project directory {project_path}: {e}")
            send_message(chat_id, f"Error: Could not create project directory. Check server permissions.")

def handle_new_project(chat_id, text, state, message_thread_id=None):
    """Handles the /new_project command."""
    parts = text.split()
    if len(parts) < 2:
        send_message(chat_id, "Usage: `/new_project <project_name> [initial_prompt]`", message_thread_id=message_thread_id)
        return
    project_name = parts[1]
    initial_prompt = " ".join(parts[2:])
    create_new_project(chat_id, project_name, state, initial_prompt, message_thread_id=message_thread_id)


def execute_file(chat_id, project_context, filename, params):
    """Executes a file in the project context with optional parameters."""
    file_path = Path(project_context) / filename
    if not file_path.is_file():
        send_message(chat_id, f"Error: File `{filename}` not found.")
        return

    command = []
    interpreter = None

    if filename.endswith('.py'):
        interpreter = sys.executable
    elif filename.endswith('.sh'):
        interpreter = 'bash'
    
    if interpreter:
        command.append(interpreter)
        command.append(str(file_path))
    elif filename.endswith(('.bat', '.cmd', '.exe')):
        command.append(str(file_path))
    else:
        send_message(chat_id, f"Error: Unsupported file type for execution: `{filename}`")
        return

    command.extend(params)

    logging.info(f"Executing command: {' '.join(command)}")
    send_message(chat_id, f"Executing: `{' '.join(command)}`")

    try:
        process = subprocess.Popen(
            command,
            cwd=project_context,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout_bytes, stderr_bytes = process.communicate(timeout=120) # 2 minute timeout
        
        try:
            stdout = stdout_bytes.decode('utf-8')
        except UnicodeDecodeError:
            stdout = stdout_bytes.decode('latin-1', errors='replace')

        try:
            stderr = stderr_bytes.decode('utf-8')
        except UnicodeDecodeError:
            stderr = stderr_bytes.decode('latin-1', errors='replace')

        # Save output to results.txt
        if stdout or stderr:
            results_file_path = Path(project_context) / "results.txt"
            try:
                with open(results_file_path, 'w', encoding='utf-8') as f:
                    if stdout:
                        f.write("--- STDOUT ---\n")
                        f.write(stdout)
                        f.write("\n")
                    if stderr:
                        f.write("--- STDERR ---\n")
                        f.write(stderr)
                logging.info(f"Execution output saved to {results_file_path}")
            except IOError as e:
                logging.error(f"Failed to write to results.txt: {e}")

        output = ""
        if stdout:
            output += f"*Output:*\n```\n{stdout.strip()}\n```\n"
        if stderr:
            output += f"*Errors:*\n```\n{stderr.strip()}\n```\n"
        
        if not output:
            output = f"`{filename}` executed with no output."
        elif stdout or stderr: # only add if there was output
            output += "\n_Output also saved to `results.txt`_"
            
        send_message(chat_id, output)

    except subprocess.TimeoutExpired:
        process.kill()
        send_message(chat_id, f"Error: Execution of `{filename}` timed out after 2 minutes.")
    except Exception as e:
        error_message = f"An error occurred while executing `{filename}`: {e}"
        logging.error(error_message)
        send_message(chat_id, error_message)


def handle_e_command(chat_id, state):
    """Handles the /e command to select a file for execution."""
    project_context = state["contexts"].get(str(chat_id))
    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.")
        return

    try:
        files = [f for f in os.listdir(project_context) if os.path.isfile(os.path.join(project_context, f))]
        if not files:
            send_message(chat_id, "No files found in the current project.")
            return

        keyboard = {
            "inline_keyboard": [[{"text": f"{get_file_icon(f)} {f}", "callback_data": f"e_select:{f}"}] for f in files]
        }
        payload = {
            'chat_id': chat_id,
            'text': "Select a file to view and execute:",
            'reply_markup': json.dumps(keyboard)
        }
        thread_id = LAST_THREAD_ID.get(str(chat_id))
        if thread_id is not None:
            payload['message_thread_id'] = thread_id
        response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error creating file list for /e command: {e}")
        send_message(chat_id, "An error occurred while listing files.")


def handle_get_file(chat_id, text, state):
    """Handles the /file command."""
    parts = text.split()
    project_context = state["contexts"].get(str(chat_id))

    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.")
        return

    if len(parts) < 2:
        # No filename provided, show file buttons
        try:
            files = [f for f in os.listdir(project_context) if os.path.isfile(os.path.join(project_context, f))]
            if not files:
                send_message(chat_id, "No files found in the current project.")
                return

            keyboard = {
                "inline_keyboard": [[{"text": f"{get_file_icon(f)} {f}", "callback_data": f"file:{f}"}] for f in files]
            }
            payload = {
                'chat_id': chat_id,
                'text': "Select a file to view:",
                'reply_markup': json.dumps(keyboard)
            }
            thread_id = LAST_THREAD_ID.get(str(chat_id))
            if thread_id is not None:
                payload['message_thread_id'] = thread_id
            response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Error creating file list: {e}")
            send_message(chat_id, "An error occurred while listing files.")
        return

    filename = parts[1]
    file_path = Path(project_context) / filename
    if file_path.is_file():
        send_file_with_content(chat_id, file_path)
    else:
        send_message(chat_id, f"Error: File `{filename}` not found in the current project.")

def handle_download_project(chat_id, state):
    """Handles the /d command to download the project as a zip file."""
    project_context = state["contexts"].get(str(chat_id))
    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.")
        return

    project_path = Path(project_context)
    project_name = project_path.name
    archive_path = Path(PROJECTS_DIR) / f"{project_name}.zip"

    try:
        shutil.make_archive(str(archive_path.with_suffix('')), 'zip', str(project_path))
        send_message(chat_id, f"Compressing `{project_name}`...")
        send_file(chat_id, str(archive_path))
        os.remove(archive_path)
    except Exception as e:
        logging.error(f"Error creating project archive: {e}")
        send_message(chat_id, f"An error occurred while creating the project archive: {e}")

def handle_kill_processes(chat_id):
    """Handles the /k command to kill gemini and node processes."""
    killed_processes = []
    errors = []

    if sys.platform == "win32":
        processes_to_kill = ["gemini.exe", "node.exe"]
        for process in processes_to_kill:
            try:
                result = subprocess.run(f"taskkill /F /IM {process}", capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    killed_processes.append(process)
                else:
                    if "not found" not in result.stderr:
                        errors.append(f"Error killing {process}: {result.stderr}")
            except Exception as e:
                errors.append(f"Error killing {process}: {e}")
    else:  # Linux and macOS
        processes_to_kill = ["gemini", "node"]
        for process in processes_to_kill:
            try:
                # Use pkill to find and kill processes by name
                result = subprocess.run(f"pkill -f {process}", capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    killed_processes.append(process)
                else:
                    # pkill returns 1 if no process is found, which is not an error in this case.
                    if result.returncode != 1:
                        errors.append(f"Error killing {process}: {result.stderr}")
            except Exception as e:
                errors.append(f"Error killing {process}: {e}")

    response_message = ""
    if killed_processes:
        response_message += f"Successfully killed: `{', '.join(killed_processes)}`\n"
    if errors:
        response_message += f"Errors: \n`{' '.join(errors)}`\n"
    if not killed_processes and not errors:
        response_message = "No running gemini or node processes found to kill."

    send_message(chat_id, response_message)

def handle_clear_session(chat_id, state):
    """Handles the /clear command to delete all sessions for the current project using the CLI."""
    project_context = state["contexts"].get(str(chat_id))
    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.")
        return

    send_message(chat_id, "🧹 Scanning for sessions to clear...")
    
    gemini_executable = shutil.which("gemini")
    if not gemini_executable:
        send_message(chat_id, "Error: `gemini` executable not found.")
        return

    try:
        # List sessions
        result = subprocess.run(
            [gemini_executable, "--list-sessions"],
            cwd=project_context,
            capture_output=True,
            text=True
        )
        
        # Parse output for session UUIDs (format: [UUID])
        # Regex to match UUIDs inside brackets
        session_ids = re.findall(r'\[([0-9a-fA-F-]{36})\]', result.stdout)
        
        if not session_ids:
             send_message(chat_id, "ℹ️ No active sessions found to clear.")
             return
             
        count = 0
        for sess_id in session_ids:
             del_res = subprocess.run(
                 [gemini_executable, "--delete-session", sess_id],
                 cwd=project_context,
                 capture_output=True,
                 text=True
             )
             if del_res.returncode == 0:
                 count += 1
             else:
                 logging.warning(f"Failed to delete session {sess_id}: {del_res.stderr}")

        if count > 0:
            send_message(chat_id, f"🧹 **Session history deleted!**\nCleared {count} old session(s).")
        else:
            send_message(chat_id, "ℹ️ No active sessions found to delete.")
            
    except Exception as e:
        error_msg = f"❌ Error clearing sessions: {e}"
        logging.error(error_msg)
        send_message(chat_id, error_msg)

def handle_new_command(chat_id, state):
    """Handles the /new command to force a fresh session for the next message."""
    state.setdefault("force_new_session", {})[chat_id] = True
    save_state(state)
    send_message(chat_id, "✨ **Ready!**\nYour next message will start a brand new conversation context (old sessions are preserved).")


def send_file_with_content(chat_id, file_path, message_thread_id=None):
    """Sends the file content and as a file attachment."""
    try:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Check if it's a Markdown file
        if file_path.suffix.lower() == '.md':
            parse_mode = "Markdown"
            # For Markdown, we don't need to wrap it in code blocks.
            # We just send the raw content, chunked.
            max_len = 4096  # Max length for a Telegram message
            for i in range(0, len(content), max_len):
                chunk = content[i:i + max_len]
                send_message(chat_id, chunk, parse_mode, message_thread_id=message_thread_id)
        else:
            # For other files, wrap in a code block
            parse_mode = "HTML"
            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            max_len = 4080  # Max length for content inside <pre><code> tags
            for i in range(0, len(escaped_content), max_len):
                chunk = escaped_content[i:i + max_len]
                send_message(chat_id, f"<pre><code>{chunk}</code></pre>", parse_mode, message_thread_id=message_thread_id)

        # Always send the file as an attachment as well
        send_file(chat_id, file_path, message_thread_id=message_thread_id)
    except Exception as e:
        logging.error(f"Error sending file with content: {e}")
        # If reading fails, just send the file as an attachment
        send_file(chat_id, file_path, message_thread_id=message_thread_id)

def handle_callback_query(callback_query, state):
    """Handles callback queries from inline keyboards."""
    callback_id = callback_query['id']
    chat_id = str(callback_query['message']['chat']['id'])
    thread_id = callback_query.get('message', {}).get('message_thread_id')
    if thread_id is not None:
        LAST_THREAD_ID[chat_id] = thread_id
    data = callback_query['data']
    message_id = callback_query['message']['message_id']

    if data.startswith("file:"):
        filename = data.split(":", 1)[1]
        project_context = state["contexts"].get(chat_id)
        if project_context:
            file_path = Path(project_context) / filename
            if file_path.is_file():
                send_file_with_content(chat_id, file_path, message_thread_id=thread_id)
            else:
                send_message(chat_id, f"Error: File `{filename}` no longer exists.", message_thread_id=thread_id)
        else:
            send_message(chat_id, "Error: Project context not found.", message_thread_id=thread_id)
    elif data.startswith("set_project:"):
        project_name = data.split(":", 1)[1]
        success, message = set_project(chat_id, project_name, state, message_thread_id=thread_id)
        
        # Edit the original message (which had the buttons)
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': message, # Use the message from set_project
            'parse_mode': 'Markdown'
        }
        requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload)
    elif data == "new_project_prompt":
        state.setdefault("awaiting_input", {})[chat_id] = "new_project_name"
        
        # Edit the original message to ask for project name
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': "Please enter the name for the new project:",
        }
        requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload)
    elif data.startswith("e_select:"):
        filename = data.split(":", 1)[1]
        project_context = state["contexts"].get(chat_id)
        if project_context:
            file_path = Path(project_context) / filename
            if file_path.is_file():
                send_file_with_content(chat_id, file_path, message_thread_id=thread_id)
                
                keyboard = {
                    "inline_keyboard": [
                        [
                            {"text": "✅ Yes", "callback_data": f"e_params_yes:{filename}"},
                            {"text": "❌ No", "callback_data": f"e_params_no:{filename}"}
                        ]
                    ]
                }
                payload = {
                    'chat_id': chat_id,
                    'text': "Would you like to pass parameters?",
                    'reply_markup': json.dumps(keyboard)
                }
                if thread_id is not None:
                    payload['message_thread_id'] = thread_id
                requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload)
            else:
                send_message(chat_id, f"Error: File `{filename}` no longer exists.", message_thread_id=thread_id)
        else:
            send_message(chat_id, "Error: Project context not found.", message_thread_id=thread_id)
        
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': f"Selected file: `{filename}`",
            'parse_mode': 'Markdown'
        }
        requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload)
    elif data.startswith("e_params_no:"):
        filename = data.split(":", 1)[1]
        project_context = state["contexts"].get(chat_id)
        if project_context:
            execute_file(chat_id, project_context, filename, [])
        else:
            send_message(chat_id, "Error: Project context not found.", message_thread_id=thread_id)
        
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': f"Executing `{filename}` without parameters...",
            'parse_mode': 'Markdown'
        }
        requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload)
    elif data.startswith("e_params_yes:"):
        filename = data.split(":", 1)[1]
        state.setdefault("awaiting_input", {})[chat_id] = f"e_exec_params:{filename}"
        
        payload = {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': (
                "To pass multiple parameters, separate them with spaces. "
                "To include spaces within a single parameter, enclose it in double quotes "
                '(e.g., `param1 "parameter two"`).\n\n'
                f"Please reply with the parameters for `{filename}`:"
            ),
            'parse_mode': 'Markdown'
        }
        requests.post(f"{TELEGRAM_API_URL}/editMessageText", data=payload)
    
    # Answer the callback query to remove the "loading" state
    requests.post(f"{TELEGRAM_API_URL}/answerCallbackQuery", data={'callback_query_id': callback_id})


def update_gemini_md(project_path, user_request=None, agent_response=None):
    """Appends user requirements and agent suggestions to GEMINI.md."""
    gemini_md_path = Path(project_path) / "GEMINI.md"
    try:
        # Ensure the file has a main header
        if not gemini_md_path.exists() or gemini_md_path.stat().st_size == 0:
            gemini_md_path.write_text("# Project Requirements\n\n", encoding='utf-8')

        with open(gemini_md_path, 'a', encoding='utf-8') as f:
            if user_request:
                logging.info(f"Appending user requirement to {gemini_md_path}")
                f.write(f"\n---\n\n### User Requirement\n\n> {user_request}\n")
            if agent_response:
                # Avoid logging empty or trivial responses
                if agent_response.strip() and "_Gemini CLI returned an empty response._" not in agent_response:
                    logging.info(f"Appending agent suggestion to {gemini_md_path}")
                    f.write(f"\n### Accepted Agent Suggestion\n\n```text\n{agent_response.strip()}\n```\n")
    except IOError as e:
        logging.error(f"Could not write to {gemini_md_path}: {e}")

def run_gemini_streaming(chat_id, command, project_context, state, message_thread_id=None):
    """Executes Gemini CLI and streams the output to Telegram."""
    logging.info(f"Executing Gemini CLI (Streaming): {' '.join(command)}")

    stream_mode = STREAM_MODE if STREAM_MODE in {"partial", "block", "off"} else "partial"
    if stream_mode != STREAM_MODE:
        logging.warning(f"Invalid STREAM_MODE '{STREAM_MODE}', falling back to 'partial'.")
    
    # Send initial "Thinking..." message
    message_id = send_message_with_id(
        chat_id,
        "_Gemini is thinking..._",
        parse_mode="Markdown",
        message_thread_id=message_thread_id
    )
    if not message_id:
        logging.error("Failed to send initial message for streaming.")
        return

    process = subprocess.Popen(
        command,
        cwd=project_context,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        bufsize=1  # Line-buffered
    )
    logging.info(f"Gemini process started with PID: {process.pid}")

    full_output = ""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    last_update_time = time.time()
    last_sent_text = ""
    update_count = 0

    stdout_clean = ""
    block_buffer = ""
    stderr_output = ""

    queue = Queue()
    stdout_done = False
    stderr_done = False

    def stream_reader(stream, label):
        for chunk in iter(lambda: stream.read(512), ''):
            if not chunk:
                break
            queue.put((label, chunk))
        queue.put((label, None))

    stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, "stdout"))
    stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, "stderr"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    def maybe_update_partial(now):
        nonlocal last_update_time, last_sent_text, update_count
        if not stdout_clean:
            return
        if now - last_update_time < STREAM_UPDATE_INTERVAL:
            return
        visible = stdout_clean
        if len(visible) > STREAM_TAIL_LIMIT:
            visible = "…" + visible[-STREAM_TAIL_LIMIT:]
        text = visible + STREAM_CURSOR
        if text != last_sent_text and edit_message_text(chat_id, message_id, text):
            last_sent_text = text
            last_update_time = now
            update_count += 1

    def maybe_update_block(now, force_flush=False):
        nonlocal last_update_time, last_sent_text, update_count, block_buffer, message_id
        # Flush full chunks immediately
        while len(block_buffer) >= STREAM_MAX_CHARS:
            chunk = block_buffer[:STREAM_MAX_CHARS]
            block_buffer = block_buffer[STREAM_MAX_CHARS:]
            if edit_message_text(chat_id, message_id, chunk):
                last_sent_text = chunk
                last_update_time = now
                update_count += 1
            if process.poll() is None or block_buffer:
                new_id = send_message_raw(chat_id, "…", message_thread_id=message_thread_id)
                if new_id:
                    message_id = new_id

        if force_flush:
            if block_buffer:
                if edit_message_text(chat_id, message_id, block_buffer):
                    last_sent_text = block_buffer
                    last_update_time = now
                    update_count += 1
            return

        if len(block_buffer) >= STREAM_MIN_CHARS and now - last_update_time >= STREAM_UPDATE_INTERVAL:
            text = block_buffer + STREAM_CURSOR
            if text != last_sent_text and edit_message_text(chat_id, message_id, text):
                last_sent_text = text
                last_update_time = now
                update_count += 1

    try:
        while True:
            try:
                label, chunk = queue.get(timeout=0.1)
            except Empty:
                label = None
                chunk = None

            if label == "stdout":
                if chunk is None:
                    stdout_done = True
                else:
                    full_output += chunk
                    clean_chunk = ansi_escape.sub('', chunk)
                    stdout_clean += clean_chunk
                    block_buffer += clean_chunk
            elif label == "stderr":
                if chunk is None:
                    stderr_done = True
                else:
                    stderr_output += chunk

            now = time.time()
            if stream_mode == "partial":
                maybe_update_partial(now)
            elif stream_mode == "block":
                maybe_update_block(now)

            if stdout_done and stderr_done and process.poll() is not None:
                break

        logging.info(f"Stream finished. Total length: {len(full_output)} chars. Total updates: {update_count}")

        if stderr_output:
            full_output += f"\n\n--- STDERR ---\n{stderr_output}"

        # Final Cleanup and Formatting
        clean_output = ansi_escape.sub('', full_output)
        
        # Save to logs
        conversation_log_path = Path(project_context) / "project_conversation.log"
        try:
            with open(conversation_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- AGENT DECISION ---\n{full_output}\n")
        except IOError as e:
            logging.error(f"Could not write agent decision to log: {e}")

        update_gemini_md(project_context, agent_response=clean_output)

        if stream_mode == "block":
            maybe_update_block(time.time(), force_flush=True)
            if stderr_output.strip():
                send_message(chat_id, f"--- STDERR ---\n{stderr_output}", "", message_thread_id=message_thread_id)
            return

        # Final Message Update with proper formatting
        final_text = clean_output.strip()
        if not final_text:
            final_text = "_Gemini CLI returned an empty response._"

        # Apply Telegram Markdown formatting
        formatted_final = format_for_telegram_paragraphs(break_sentences_into_lines(final_text))
        
        # If output is too long, we might need to split it, but editMessageText can't split.
        # So we edit the first part and send new messages for the rest.
        max_len = 4096
        parts = [formatted_final[i:i + max_len] for i in range(0, len(formatted_final), max_len)]

        if parts:
            # Update the existing message with the first part
            try:
                res_ok = edit_message_text(chat_id, message_id, parts[0], parse_mode="Markdown")
                if not res_ok:
                    raise Exception("API Error: editMessageText failed")
            except Exception as e:
                logging.warning(f"Markdown send failed, falling back to plain text: {e}")
                # Fallback: Try sending without Parse Mode (Plain Text)
                edit_message_text(chat_id, message_id, parts[0])

            # Send remaining parts as new messages
            for part in parts[1:]:
                try:
                    send_message(chat_id, part, "Markdown", message_thread_id=message_thread_id)
                except:
                    send_message(chat_id, part, "", message_thread_id=message_thread_id) # Fallback for remaining parts too

        # Check for files to display
        match = re.search(r'`([^`\n]+)`', clean_output)
        if match:
            filename = match.group(1)
            file_path = Path(project_context) / filename
            if file_path.is_file():
                send_file_with_content(chat_id, file_path, message_thread_id=message_thread_id)

    except Exception as e:
        logging.error(f"Error in streaming: {e}")
        send_message(chat_id, f"Error during execution: {e}", message_thread_id=message_thread_id)


def handle_gemini_prompt(chat_id, text, state, message_thread_id=None):
    """Handles a regular message as a prompt to Gemini CLI."""
    project_context = state["contexts"].get(str(chat_id))
    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.", message_thread_id=message_thread_id)
        return

    # Increment and check the prompt counter
    prompt_counter = state["prompt_counters"].get(project_context, 0) + 1
    state["prompt_counters"][project_context] = prompt_counter
    
    update_gemini_md(project_context, user_request=text)

    # send_message(chat_id, f"Processing your request in project `{project_context}`...") # Removed to reduce noise
    conversation_log_path = Path(project_context) / "project_conversation.log"

    try:
        with open(conversation_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- USER REQUEST ---\n{text}\n")
    except IOError as e:
        logging.error(f"Could not write to {conversation_log_path}: {e}")

    # Prepare Gemini command
    gemini_executable = shutil.which("gemini")
    if not gemini_executable:
        send_message(chat_id, "Error: `gemini` command not found.")
        return

    command = [gemini_executable, "--yolo"]
    
    # Check if we should force a new session
    force_new = state.get("force_new_session", {}).get(str(chat_id), False)
    
    if force_new:
        logging.info("Force new session flag detected. Starting fresh session.")
        # Consume the flag
        if str(chat_id) in state.get("force_new_session", {}):
            del state["force_new_session"][str(chat_id)]
            save_state(state)
    else:
        # Check if a session exists using CLI
        try:
            check_res = subprocess.run(
                [gemini_executable, "--list-sessions"],
                cwd=project_context,
                capture_output=True,
                text=True
            )
            # If output contains at least one bracketed UUID, assume sessions exist
            if "[" in check_res.stdout and "]" in check_res.stdout:
                 command.extend(["--resume", "latest"])
            else:
                 logging.info(f"No active sessions found (CLI check), starting fresh.")
        except Exception as e:
            logging.warning(f"Failed to check sessions: {e}. Defaulting to fresh session.")

    command.extend(["--prompt", text])
    
    if DEBUG_MODE:
        command.append("--debug")

    # Start streaming in a separate thread
    thread = threading.Thread(
        target=run_gemini_streaming,
        args=(chat_id, command, project_context, state, message_thread_id)
    )
    thread.daemon = True
    thread.start()

    # Send periodic reminder
    if prompt_counter % 5 == 0:
        reminder_message = (
            f"You've sent {prompt_counter} requests for this project. To keep the requirements concise, "
            f"you may want to refine the context soon using the `/context` command."
        )
        send_message(chat_id, reminder_message, message_thread_id=message_thread_id)


# --- File System Observer ---

file_observers = {}


class ProjectFileHandler(FileSystemEventHandler):
    """Handles file system events and sends notifications to Telegram."""
    def __init__(self, chat_id, project_path):
        self.chat_id = chat_id
        self.project_path = Path(project_path)
        # Use strings for parts checking, as it's more reliable across paths
        self.ignore_patterns = ["venv", "__pycache__"]

    def _should_ignore(self, event_path):
        """Checks if the event path should be ignored."""
        try:
            path = Path(event_path)
            # Check if any part of the path matches an ignore pattern
            return any(part in self.ignore_patterns for part in path.parts)
        except TypeError:
            return False

    def on_any_event(self, event):
        """Callback for any file system event."""
        if self._should_ignore(event.src_path):
            return
        if hasattr(event, 'dest_path') and self._should_ignore(event.dest_path):
            return

        # Ignore noisy directory modification events
        if event.is_directory and event.event_type == 'modified':
            return

        try:
            event_type_map = {
                'created': 'Created',
                'deleted': 'Deleted',
                'modified': 'Modified',
                'moved': 'Moved/Renamed'
            }
            event_type = event_type_map.get(event.event_type, 'Changed')
            path_type = "directory" if event.is_directory else "file"

            message = f"ℹ️ *Project Update*\n"
            if event.event_type == 'moved':
                src = Path(event.src_path).relative_to(self.project_path)
                dest = Path(event.dest_path).relative_to(self.project_path)
                message += f"_{event_type}_ {path_type}:\n`{src}` ➡️ `{dest}`"
            else:
                path = Path(event.src_path).relative_to(self.project_path)
                message += f"_{event_type}_ {path_type}: `{path}`"

            send_message(self.chat_id, message)
            logging.info(f"Sent file system notification to {self.chat_id}: {message}")

        except Exception as e:
            logging.error(f"Error processing file system event: {e}", exc_info=True)


def start_file_observer(chat_id, project_path):
    """Starts a file system observer for a given project path."""
    stop_file_observer(chat_id)  # Ensure any existing observer is stopped first

    event_handler = ProjectFileHandler(chat_id, project_path)
    observer = Observer()
    observer.schedule(event_handler, project_path, recursive=True)

    # Run observer in a separate daemon thread
    observer_thread = threading.Thread(target=observer.start)
    observer_thread.daemon = True
    observer_thread.start()

    file_observers[chat_id] = observer
    logging.info(f"Started file system observer for chat {chat_id} on path: {project_path}")


def stop_file_observer(chat_id):
    """Stops the file system observer for a given chat ID."""
    if chat_id in file_observers:
        observer = file_observers.pop(chat_id)
        if observer.is_alive():
            observer.stop()
            observer.join()  # Wait for the thread to terminate
        logging.info(f"Stopped file system observer for chat {chat_id}")




def main():
    """The main function to run the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logging.critical("TELEGRAM_BOT_TOKEN not found in environment variables. Please set it in a .env file.")
        sys.exit(1)

    logging.info("=================================================")
    logging.info("    Starting Gemini-CLI Telegram Bot...")
    logging.info("=================================================")
    logging.info(f"Configuration:")
    logging.info(f" - PROJECTS_DIR: {PROJECTS_DIR}")
    logging.info(f" - AUTHORIZED_USER_ID: {AUTHORIZED_USER_ID}")

    state = load_state()

    while True:
        try:
            offset = state.get("last_update_id", 0) + 1
            logging.debug(f"Fetching updates with offset: {offset}")
            response = requests.get(
                f"{TELEGRAM_API_URL}/getUpdates",
                params={'offset': offset, 'timeout': 1},
                timeout=10
            )
            response.raise_for_status()
            updates = response.json()

            if not updates.get("ok"):
                logging.error(f"Error fetching updates from Telegram API: {updates}")
                time.sleep(10)
                continue

            for update in updates.get("result", []):
                logging.info(f"Processing raw update object: {update}")
                update_id = update['update_id']
                state["last_update_id"] = update_id # Process one by one

                if 'callback_query' in update:
                    handle_callback_query(update['callback_query'], state)
                    state["last_update_id"] = update['update_id']
                    save_state(state)
                    continue

                if 'message' not in update:
                    continue
                
                message = update['message']
                chat_id = str(message['chat']['id'])
                thread_id = message.get('message_thread_id')
                if thread_id is not None:
                    LAST_THREAD_ID[chat_id] = thread_id
                
                # --- Authorization Check ---
                if chat_id != AUTHORIZED_USER_ID:
                    logging.warning(f"Unauthorized access attempt from Chat ID: {chat_id}")
                    send_message(chat_id, "_You are not authorized to use this bot._", message_thread_id=thread_id)
                    continue

                # --- Context Workflow Handler ---
                if chat_id in state.get("context_workflows", {}):
                    # Handle file upload
                    if 'document' in message:
                        doc = message['document']
                        if doc.get('file_name', '').lower() == 'gemini.md':
                            project_context = state["contexts"].get(chat_id)
                            gemini_md_path = Path(project_context) / "GEMINI.md"
                            
                            file_info_res = requests.get(f"{TELEGRAM_API_URL}/getFile", params={'file_id': doc['file_id']})
                            file_info = file_info_res.json()['result']
                            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['file_path']}"
                            
                            download_res = requests.get(file_url)
                            gemini_md_path.write_bytes(download_res.content)
                            
                            send_message(chat_id, "Successfully updated `GEMINI.md` with your uploaded file.")
                            del state["context_workflows"][chat_id]
                        else:
                            send_message(chat_id, "File ignored. Please upload a file named `GEMINI.md` to proceed.")
                        save_state(state)
                        continue

                    # Handle text responses for workflow
                    text = message.get('text', '').lower()
                    if text in ["1", "accept"]:
                        project_context = state["contexts"][chat_id]
                        proposed_text = state["context_workflows"][chat_id]["proposed_text"]
                        (Path(project_context) / "GEMINI.md").write_text(proposed_text, encoding='utf-8')
                        send_message(chat_id, "Project context (`GEMINI.md`) has been successfully updated.")
                        del state["context_workflows"][chat_id]
                    elif text in ["3", "decline"]:
                        send_message(chat_id, "Operation cancelled. No changes have been made.")
                        del state["context_workflows"][chat_id]
                    else: # Option 2: Suggest Edits
                        send_message(chat_id, "_Incorporating your edits and generating a new proposal..._")
                        project_context = state["contexts"][chat_id]
                        proposed_text = state["context_workflows"][chat_id]["proposed_text"]
                        user_edits = message.get('text', '')
                        
                        prompt = (
                            "The user has suggested edits to the proposed requirements. Please incorporate the following "
                            f"feedback and generate a new, updated version. User Feedback: '{user_edits}'. "
                            f"Previous Proposal:\n---\n{proposed_text}"
                        )
                        
                        gemini_executable = shutil.which("gemini")
                        if not gemini_executable:
                            send_message(chat_id, "Error: `gemini` command not found. Is gemini-cli installed and in the system's PATH?")
                            return
                        
                        command = [gemini_executable, "--yolo", "--prompt", prompt]
                        result = subprocess.run(command, cwd=project_context, capture_output=True, text=True, timeout=300, check=True)
                        
                        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        new_proposal = ansi_escape.sub('', result.stdout)

                        state["context_workflows"][chat_id]["proposed_text"] = new_proposal
                        
                        send_message(chat_id, "*Agent's New Proposed Update for `GEMINI.md`*")
                        escaped_proposal = new_proposal.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        send_message(chat_id, f"<pre><code>{escaped_proposal}</code></pre>", "HTML")
                        send_message(chat_id, "You can now Accept (1), Decline (3), suggest more edits, or upload a file.")

                    save_state(state)
                    continue

                if 'voice' in message:
                    handle_voice_message(message, state, message_thread_id=thread_id)
                    save_state(state)
                    continue
                    
                text = message.get('text', '')

                # --- Awaiting Input Handler ---
                awaiting_input_type = state.get("awaiting_input", {}).get(chat_id)
                if awaiting_input_type:
                    if not text.startswith('/'):
                        del state["awaiting_input"][chat_id]  # Consume it
                        if awaiting_input_type == "new_project_name":
                            project_name = text.strip()
                            if project_name:
                                create_new_project(chat_id, project_name, state, message_thread_id=thread_id)
                            else:
                                send_message(chat_id, "Invalid project name. Operation cancelled.")
                            save_state(state)
                            continue
                        elif awaiting_input_type.startswith("exec_params:"):
                            filename = awaiting_input_type.split(":", 1)[1]
                            params = shlex.split(text.strip())
                            project_context = state["contexts"].get(chat_id)
                            if project_context:
                                execute_file(chat_id, project_context, filename, params)
                            else:
                                send_message(chat_id, "Error: Project context not found.")
                            save_state(state)
                            continue
                        elif awaiting_input_type.startswith("e_exec_params:"):
                            filename = awaiting_input_type.split(":", 1)[1]
                            params = shlex.split(text.strip())
                            project_context = state["contexts"].get(chat_id)
                            if project_context:
                                execute_file(chat_id, project_context, filename, params)
                            else:
                                send_message(chat_id, "Error: Project context not found.")
                            save_state(state)
                            continue
                    else:  # User sent a command, cancel awaiting input
                        del state["awaiting_input"][chat_id]
                        send_message(chat_id, "Operation cancelled.")
                        # Let it fall through to command processing

                # --- Standard Command Dispatcher ---
                if not text:
                    continue

                if text.startswith("/set_project") or text.startswith("/p"):
                    handle_set_project(chat_id, text, state, message_thread_id=thread_id)
                elif text.startswith("/new_project"):
                    handle_new_project(chat_id, text, state, message_thread_id=thread_id)
                elif text.startswith("/file") or text.startswith("/f"):
                    handle_get_file(chat_id, text, state)
                elif text.startswith("/e"):
                    handle_e_command(chat_id, state)
                elif text.startswith("/d"):
                    handle_download_project(chat_id, state)
                elif text.startswith("/k"):
                    handle_kill_processes(chat_id)
                elif text == "/clear":
                    handle_clear_session(chat_id, state)
                elif text == "/new":
                    handle_new_command(chat_id, state)
                elif text == "/context":
                    handle_context_command(chat_id, state)
                elif text == "/current_project":
                    current_project = state["contexts"].get(chat_id, "None")
                    send_message(chat_id, f"Current project is: `{current_project}`", message_thread_id=thread_id)
                else:
                    handle_gemini_prompt(chat_id, text, state, message_thread_id=thread_id)

                save_state(state)

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error during getUpdates: {e}. Retrying in 2 seconds...")
            time.sleep(2)
        except KeyboardInterrupt:
            logging.info("Bot shutting down gracefully.")
            for chat_id in list(file_observers.keys()):
                stop_file_observer(chat_id)
            break
        except Exception as e:
            logging.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
            time.sleep(10)

def handle_voice_message(message, state, message_thread_id=None):
    """Handles a voice message by transcribing it and passing it to Gemini."""
    chat_id = str(message['chat']['id'])
    project_context = state["contexts"].get(chat_id)
    if not project_context:
        send_message(chat_id, "No project context set. Please use `/set_project <project_name>` first.", message_thread_id=message_thread_id)
        return

    voice = message['voice']
    file_id = voice['file_id']
    
    send_message(chat_id, "_Transcribing voice message..._", message_thread_id=message_thread_id)

    try:
        # Get file path from Telegram
        file_info_res = requests.get(f"{TELEGRAM_API_URL}/getFile", params={'file_id': file_id})
        file_info_res.raise_for_status()
        file_info = file_info_res.json()['result']
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['file_path']}"
        
        # Download the voice file using streaming to ensure it's complete
        voice_content = bytearray()
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                voice_content.extend(chunk)

        # Save a copy of the voice file
        voice_dir = Path(project_context) / "voice"
        voice_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Get file extension from telegram file_path, default to .ogg
        file_extension = Path(file_info['file_path']).suffix or '.ogg'
        voice_filename = f"{timestamp}{file_extension}"
        voice_filepath = voice_dir / voice_filename
        with open(voice_filepath, 'wb') as f:
            f.write(voice_content)
        logging.info(f"Saved voice message to: {voice_filepath}")

        # Transcribe using Google Speech-to-Text
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=bytes(voice_content))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US",
        )
        
        response = client.recognize(config=config, audio=audio)
        
        if not response.results or not response.results[0].alternatives:
            send_message(chat_id, "Could not understand the audio. Please try again.", message_thread_id=message_thread_id)
            return

        transcript = response.results[0].alternatives[0].transcript
        
        # Send transcript to user and process as a prompt
        send_message(chat_id, f"Heard: \"_{transcript}_\"", message_thread_id=message_thread_id)
        handle_gemini_prompt(chat_id, transcript, state, message_thread_id=message_thread_id)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading voice file: {e}")
        send_message(chat_id, "Error downloading voice file for transcription.", message_thread_id=message_thread_id)
    except Exception as e:
        logging.error(f"An error occurred during speech-to-text: {e}", exc_info=True)
        send_message(chat_id, "An error occurred during transcription.", message_thread_id=message_thread_id)

if __name__ == "__main__":
    main()
