#!/usr/bin/env bash
set -euo pipefail

echo "[devcontainer] Setting up pip virtual environments..."

# Resolve a Python binary
resolve_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  elif [ -x "/opt/conda/bin/python" ]; then
    echo "/opt/conda/bin/python"
  else
    echo "[ERROR] No python found on PATH (tried python3, python, /opt/conda/bin/python)" >&2
    exit 1
  fi
}

PYBIN="$(resolve_python)"
VENV_ROOT="${HOME}/.venvs"
DEFAULT_ENV="pibob-dev"   # change if you want a different default

mkdir -p "${VENV_ROOT}"

create_venv() {
  local name="$1"
  local req_file="requirements/${name}.txt"
  local venv_path="${VENV_ROOT}/${name}"

  if [ ! -d "${venv_path}" ]; then
    echo ">>> Creating venv: ${name}"
    "${PYBIN}" -m venv "${venv_path}"
    "${venv_path}/bin/pip" install -U pip

    if [ -f "${req_file}" ]; then
      echo ">>> Installing requirements from ${req_file}"
      "${venv_path}/bin/pip" install --default-timeout=300 -r "${req_file}"
    else
      echo ">>> No ${req_file} found, skipping requirements"
    fi

    # 💡 Add editable install for the main repo
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
      echo ">>> Installing current repo in editable mode (-e .)"
      "${venv_path}/bin/pip" install -e .
    else
      echo ">>> No setup.py or pyproject.toml found — skipping editable install"
    fi
  else
    echo ">>> ${name} already exists, skipping creation"
  fi
}

create_venv kawin-env
create_venv kawin-envold
create_venv pibo-dev

echo
echo "[devcontainer] Available virtual environments in ${VENV_ROOT}:"
ls -1 "${VENV_ROOT}" || true

# Make DEFAULT_ENV the default in interactive shells (bash & zsh)
ACTIVATE_LINE="source ${VENV_ROOT}/${DEFAULT_ENV}/bin/activate"
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
  if [ -f "$rc" ] && ! grep -qxF "${ACTIVATE_LINE}" "$rc" 2>/dev/null; then
    echo "${ACTIVATE_LINE}" >> "$rc"
  fi
done

echo "[devcontainer] Default environment set to ${DEFAULT_ENV}"
echo "[devcontainer] Python used to create venvs: ${PYBIN}"
