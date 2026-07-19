"""Determinism, device, and console-progress helpers."""
import os
import sys
import time
import random
import numpy as np
import torch


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Console progress state.
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for _ in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for _ in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)

    term_width = 80
    for _ in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')
    for _ in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secs = int(seconds)
    millis = int((seconds - secs) * 1000)

    parts = []
    if days > 0: parts.append(f'{days}D')
    if hours > 0 and len(parts) < 2: parts.append(f'{hours}h')
    if minutes > 0 and len(parts) < 2: parts.append(f'{minutes}m')
    if secs > 0 and len(parts) < 2: parts.append(f'{secs}s')
    if millis > 0 and len(parts) < 2: parts.append(f'{millis}ms')
    return ''.join(parts) if parts else '0ms'
