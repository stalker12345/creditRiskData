#!/bin/bash

# Скрипт для запуска Streamlit приложения
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Запуск системы оценки кредитного риска..."
echo ""

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "🛠 Виртуальное окружение не найдено, создаю его..."
    python3 -m venv venv
fi

# Активация виртуального окружения
source venv/bin/activate

# Проверка установки streamlit
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Установка зависимостей..."
    python -m pip install -r streamlit_model/requirements_streamlit.txt
fi

# Проверка наличия модели
if [ ! -f "streamlit_model/model.skops" ]; then
    echo "❌ Модель не найдена в streamlit_model/model.skops"
    echo "Убедитесь, что модель была сохранена!"
    exit 1
fi

echo "✅ Все проверки пройдены!"
echo ""
echo "🌐 Запуск приложения..."
echo "Приложение откроется в браузере по адресу: http://localhost:8501"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""

# Запуск Streamlit
python -m streamlit run app.py
