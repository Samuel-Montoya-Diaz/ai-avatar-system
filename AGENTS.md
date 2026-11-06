# Contexto
TFG de la Universidad de La Laguna: reconstrucción histórica virtual de
San Cristóbal de La Laguna (s. XVI). Basado en ai-avatar-system (MIT).
Objetivo: NPCs históricos con los que hablar en lenguaje natural
(LLM + TTS + lip-sync).

# Entorno
Windows 11 + WSL2 (Ubuntu), Docker Desktop, GPU NVIDIA RTX 5070
(Blackwell, 12 GB VRAM, requiere CUDA 12.8+).
Proyecto en ~/TFG/ai-avatar-system dentro de WSL.

# Stack
Backend FastAPI (backend/), frontend Next.js (frontend/),
Postgres, Redis, Celery.

# Reglas
- Trabaja siempre en la rama actual, nunca en main.
- Antes de modificar, explica el plan y los ficheros afectados.
- Nunca leas, muestres ni edites .env ni ficheros con claves.
- No toques backend/models/ ni lo añadas a git (modelos pesados).
- Cambios pequeños, un commit por tarea, mensajes tipo "feat(backend): ...".
- Ejecuta los tests (pytest en backend/) tras cambios en backend.
- Comenta en español; código e identificadores en inglés.
