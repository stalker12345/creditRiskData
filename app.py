import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Добавляем путь к модулю загрузки модели
streamlit_model_path = Path(__file__).parent / "streamlit_model"
sys.path.insert(0, str(streamlit_model_path))

try:
    from load_model import load_model, load_metadata, load_metrics, load_boxcox_params, predict_risk
except ImportError as e:
    st.error(f"Ошибка импорта модуля загрузки модели: {str(e)}")
    st.info(f"Путь поиска: {streamlit_model_path}")
    st.info("Убедитесь, что все зависимости установлены из requirements.txt")
    st.stop()

# Настройка страницы
st.set_page_config(
    page_title="Система оценки кредитного риска",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных модели один раз
@st.cache_resource
def load_all_model_data():
    """Загружает модель, метаданные и метрики"""
    try:
        model = load_model()
        metadata = load_metadata()
        metrics = load_metrics()
        lambdas = load_boxcox_params()
        return model, metadata, metrics, lambdas
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.stop()

# Загружаем модель
model, metadata, metrics, lambdas = load_all_model_data()
income_range = metadata['feature_ranges']['person_income']
income_min = max(1, int(income_range['min']))
income_max = int(income_range['max'])
income_default = min(max(50000, income_min), income_max)
loan_percent_income_max = float(metadata['feature_ranges']['loan_percent_income']['max'])

HOME_OWNERSHIP_LABELS = {
    "MORTGAGE": "Ипотека",
    "OTHER": "Другое",
    "OWN": "Собственное жильё",
    "RENT": "Аренда",
}

LOAN_INTENT_LABELS = {
    "DEBTCONSOLIDATION": "Объединение долгов",
    "EDUCATION": "Образование",
    "HOMEIMPROVEMENT": "Ремонт жилья",
    "MEDICAL": "Медицинские расходы",
    "PERSONAL": "Личные нужды",
    "VENTURE": "Бизнес",
}

LOAN_GRADE_LABELS = {
    "A": "A - высокая",
    "B": "B - хорошая",
    "C": "C - средняя",
    "D": "D - низкая",
    "E": "E - очень низкая",
    "Other": "Другая",
}

LOAN_GRADE_ORDER = ["A", "B", "C", "D", "E", "Other"]

HOME_OWNERSHIP_OTHER_EXPLANATION = (
    "`Другое` означает статус жилья, который не относится к ипотеке, "
    "собственному жилью или аренде."
)

LOAN_GRADE_AUTO_EXPLANATION = (
    "Кредитный рейтинг заявки не нужно выбирать вручную. В демо-версии "
    "приложение рассчитывает его автоматически при отправке формы: основой "
    "служит процентная ставка, а корректировки зависят от доли дохода на "
    "кредит, прошлых дефолтов и стажа работы. В реальной банковской системе "
    "это правило нужно заменить на официальный внутренний рейтинг или "
    "переобучить модель без признака `loan_grade`."
)

LOAN_GRADE_RULE_DETAILS = """
**Базовый рейтинг по процентной ставке:**
- до 8% включительно — A
- 8.1–11% — B
- 11.1–14% — C
- 14.1–17% — D
- 17.1–21% — E
- выше 21% — Другая

**Корректировки:**
- доля дохода на кредит от 40% — рейтинг ухудшается на 1 ступень
- доля дохода на кредит до 20% — рейтинг улучшается на 1 ступень
- есть прошлые дефолты — рейтинг ухудшается на 1 ступень
- стаж меньше 1 года — рейтинг ухудшается на 1 ступень
- стаж от 5 лет — рейтинг улучшается на 1 ступень
"""

MODEL_TYPE_LABELS = {
    "GradientBoostingClassifier": "Градиентный бустинг",
}


def display_label(labels, value):
    return labels.get(value, str(value))


def estimate_loan_grade(loan_int_rate, loan_percent_income, has_previous_default, person_emp_length):
    """Оценивает loan_grade для демо-сценария, когда внешнего рейтинга нет."""
    if loan_int_rate <= 8:
        grade_index = 0
    elif loan_int_rate <= 11:
        grade_index = 1
    elif loan_int_rate <= 14:
        grade_index = 2
    elif loan_int_rate <= 17:
        grade_index = 3
    elif loan_int_rate <= 21:
        grade_index = 4
    else:
        grade_index = 5

    if loan_percent_income >= 0.4:
        grade_index += 1
    elif loan_percent_income <= 0.2:
        grade_index -= 1

    if has_previous_default:
        grade_index += 1

    if person_emp_length < 1:
        grade_index += 1
    elif person_emp_length >= 5:
        grade_index -= 1

    grade_index = max(0, min(grade_index, len(LOAN_GRADE_ORDER) - 1))
    return LOAN_GRADE_ORDER[grade_index]

# Заголовок приложения
st.title("🏦 Система оценки кредитного риска")
st.markdown("---")

# Боковая панель с информацией
with st.sidebar:
    st.header("📊 О модели")
    
    st.markdown("### Качество модели:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Точность", f"{metrics['test_accuracy']*100:.1f}%")
        st.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
    with col2:
        st.metric("F1-мера", f"{metrics['test_f1']:.3f}")
        st.metric("Точность отказов", f"{metrics['test_precision']:.3f}")
    
    st.markdown("### Важные метрики:")
    st.info(f"**Полнота:** {metrics['test_recall']*100:.1f}%\n\n"
            f"Находит {metrics['test_recall']*100:.1f}% всех дефолтов")
    
    st.markdown("### Статистика:")
    cm = metrics['confusion_matrix']
    st.write(f"✅ Правильных предсказаний: {cm['tn'] + cm['tp']}")
    st.write(f"❌ Ложных отказов: {cm['fp']}")
    st.write(f"⚠️ Пропущенных дефолтов: {cm['fn']}")
    
    st.markdown("---")
    st.markdown("### 💡 Как это работает?")
    with st.expander("Подробнее"):
        st.markdown("""
        Модель использует машинное обучение для оценки вероятности 
        дефолта по кредиту на основе:
        - Личных данных заёмщика
        - Финансовых показателей
        - Истории кредитов
        
        **Важно:** Решение принимается на основе анализа тысяч 
        реальных кредитных заявок.
        """)

# Основной контент
tab1, tab2, tab3 = st.tabs(["🎯 Оценка риска", "📚 О метриках", "ℹ️ О системе"])

with tab1:
    st.header("Введите данные заёмщика")
    st.markdown("Заполните форму ниже для оценки кредитного риска")
    
    # Создаём форму
    with st.form("credit_risk_form"):
        # Разделяем на колонки для лучшего UI
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Личные данные")
            
            person_age = st.slider(
                "Возраст заёмщика",
                min_value=int(metadata['feature_ranges']['person_age']['min']),
                max_value=int(metadata['feature_ranges']['person_age']['max']),
                value=30,
                help="Возраст заёмщика в годах"
            )
            
            person_income = st.number_input(
                "Годовой доход (в долларах)",
                min_value=income_min,
                max_value=income_max,
                value=income_default,
                step=1000,
                help="Введите годовой доход заёмщика в долларах США"
            )
            
            person_home_ownership = st.selectbox(
                "Статус жилья",
                options=metadata['person_home_ownership_categories'],
                format_func=lambda value: display_label(HOME_OWNERSHIP_LABELS, value),
                help=(
                    "Тип жилья заёмщика. "
                    "Другое — статус жилья вне категорий ипотека, собственное жильё и аренда."
                )
            )
            st.caption(HOME_OWNERSHIP_OTHER_EXPLANATION)
            
            person_emp_length = st.slider(
                "Опыт работы (лет)",
                min_value=0.0,
                max_value=float(metadata['feature_ranges']['person_emp_length']['max']),
                value=5.0,
                step=0.5,
                help="Количество лет опыта работы на текущем месте"
            )
            
            cb_person_default_on_file = st.radio(
                "Есть ли история дефолтов?",
                options=["Нет", "Да"],
                help="Были ли у заёмщика дефолты в прошлом",
                horizontal=True
            )
            
            cb_person_cred_hist_length = st.slider(
                "Длина кредитной истории (лет)",
                min_value=int(metadata['feature_ranges']['cb_person_cred_hist_length']['min']),
                max_value=int(metadata['feature_ranges']['cb_person_cred_hist_length']['max']),
                value=5,
                help="Сколько лет кредитной истории у заёмщика"
            )
        
        with col2:
            st.subheader("💵 Параметры кредита")
            
            loan_amnt = st.number_input(
                "Сумма кредита (в долларах)",
                min_value=int(metadata['feature_ranges']['loan_amnt']['min']),
                max_value=int(metadata['feature_ranges']['loan_amnt']['max']),
                value=10000,
                step=500,
                help="Запрашиваемая сумма кредита"
            )
            
            loan_int_rate = st.slider(
                "Процентная ставка (%)",
                min_value=float(metadata['feature_ranges']['loan_int_rate']['min']),
                max_value=float(metadata['feature_ranges']['loan_int_rate']['max']),
                value=10.0,
                step=0.1,
                format="%.2f",
                help="Процентная ставка по кредиту"
            )
            
            loan_percent_income = st.slider(
                "Доля дохода на кредит",
                min_value=0.0,
                max_value=loan_percent_income_max,
                value=0.2,
                step=0.01,
                format="%.2f",
                help="Какую долю дохода займёт выплата по кредиту"
            )
            
            loan_intent = st.selectbox(
                "Цель кредита",
                options=metadata['loan_intent_categories'],
                format_func=lambda value: display_label(LOAN_INTENT_LABELS, value),
                help="Для чего заёмщик планирует использовать кредит"
            )
            
            st.info(LOAN_GRADE_AUTO_EXPLANATION)
            with st.expander("Как рассчитывается кредитный рейтинг"):
                st.markdown(LOAN_GRADE_RULE_DETAILS)
        
        # Кнопка предсказания
        submitted = st.form_submit_button("🔍 Оценить риск", use_container_width=True)
    
    # Обработка формы
    if submitted:
        # Подготовка данных
        cb_default = 1 if cb_person_default_on_file == "Да" else 0
        loan_grade = estimate_loan_grade(
            loan_int_rate=loan_int_rate,
            loan_percent_income=loan_percent_income,
            has_previous_default=cb_default == 1,
            person_emp_length=person_emp_length,
        )
        
        user_data = {
            "person_age": float(person_age),
            "person_income": float(person_income),  # В исходном формате!
            "person_home_ownership": person_home_ownership,
            "person_emp_length": float(person_emp_length),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": cb_default,
            "cb_person_cred_hist_length": int(cb_person_cred_hist_length)
        }
        
        # Делаем предсказание
        try:
            result = predict_risk(user_data, model, metadata, lambdas)
            
            # Результаты
            st.markdown("---")
            st.header("📊 Результаты оценки")
            st.info(
                "Автоматически рассчитанный кредитный рейтинг заявки: "
                f"**{display_label(LOAN_GRADE_LABELS, loan_grade)}**"
            )
            
            # Основные метрики
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_prob = result['probability_default'] * 100
                if risk_prob < 30:
                    st.metric(
                        "Вероятность дефолта",
                        f"{risk_prob:.1f}%",
                        delta="- Низкий риск",
                        delta_color="normal"
                    )
                elif risk_prob < 50:
                    st.metric(
                        "Вероятность дефолта",
                        f"{risk_prob:.1f}%",
                        delta="- Умеренный риск",
                        delta_color="off"
                    )
                else:
                    st.metric(
                        "Вероятность дефолта",
                        f"{risk_prob:.1f}%",
                        delta="- Высокий риск",
                        delta_color="inverse"
                    )
            
            with col2:
                safe_prob = result['probability_non_default'] * 100
                st.metric(
                    "Вероятность возврата",
                    f"{safe_prob:.1f}%",
                )
            
            with col3:
                decision = "✅ Одобрено" if result['prediction'] == 0 else "❌ Отказ"
                st.metric(
                    "Рекомендация",
                    decision
                )
            
            # Визуализация вероятностей
            st.markdown("### 📈 Визуализация вероятностей")
            prob_df = pd.DataFrame({
                'Вероятность': ['Возврат кредита', 'Дефолт'],
                'Значение': [
                    result['probability_non_default'] * 100,
                    result['probability_default'] * 100
                ]
            })
            
            # График
            import plotly.express as px
            fig = px.bar(
                prob_df,
                x='Вероятность',
                y='Значение',
                color='Вероятность',
                color_discrete_map={
                    'Возврат кредита': '#2ecc71',
                    'Дефолт': '#e74c3c'
                },
                text='Значение',
                text_auto='.1f'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                title="Вероятности исходов",
                yaxis_title="Вероятность (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Бизнес-интерпретация
            st.markdown("---")
            st.header("💼 Бизнес-интерпретация")
            
            if result['prediction'] == 0:
                st.success("### ✅ Рекомендация: ОДОБРИТЬ КРЕДИТ")
                st.markdown(f"""
                **Обоснование:**
                - Вероятность возврата кредита: **{safe_prob:.1f}%**
                - Вероятность дефолта: **{risk_prob:.1f}%**
                
                **Риск:** Низкий риск дефолта. Заёмщик демонстрирует стабильные 
                финансовые показатели и соответствует критериям одобрения.
                
                **Экономический эффект:**
                - Ожидаемая прибыль от кредита превышает потенциальные риски
                - Клиент получает доступ к финансированию
                """)
            else:
                st.error("### ❌ Рекомендация: ОТКАЗАТЬ В КРЕДИТЕ")
                st.markdown(f"""
                **Обоснование:**
                - Вероятность дефолта: **{risk_prob:.1f}%**
                - Вероятность возврата: **{safe_prob:.1f}%**
                
                **Риск:** Высокий риск дефолта. Заёмщик не соответствует 
                критериям одобрения на основе анализа модели.
                
                **Причины повышенного риска:**
                - Модель обнаружила признаки, связанные с повышенной вероятностью дефолта
                - Рекомендуется запросить дополнительные документы или предложить альтернативные условия
                
                **Экономический эффект:**
                - Избежание потенциальных потерь от невыплаченного кредита
                """)
            
            # Факторы риска
            st.markdown("### 🔍 Ключевые факторы оценки")
            
            # Анализ введённых данных
            risk_factors = []
            
            if loan_percent_income > 0.4:
                risk_factors.append({
                    "factor": "Высокая доля дохода на кредит",
                    "value": f"{loan_percent_income*100:.1f}%",
                    "impact": "⚠️ Повышает риск"
                })
            
            if cb_default == 1:
                risk_factors.append({
                    "factor": "История дефолтов",
                    "value": "Есть",
                    "impact": "❌ Значительно повышает риск"
                })
            
            if loan_int_rate > 15:
                risk_factors.append({
                    "factor": "Высокая процентная ставка",
                    "value": f"{loan_int_rate:.2f}%",
                    "impact": "⚠️ Указывает на повышенный риск"
                })
            
            if loan_grade in ['D', 'E', 'Other']:
                risk_factors.append({
                    "factor": "Низкий кредитный рейтинг заявки",
                    "value": display_label(LOAN_GRADE_LABELS, loan_grade),
                    "impact": "⚠️ Повышает риск"
                })
            
            if person_emp_length < 2:
                risk_factors.append({
                    "factor": "Маленький опыт работы",
                    "value": f"{person_emp_length:.1f} лет",
                    "impact": "⚠️ Может указывать на нестабильность"
                })
            
            positive_factors = []
            
            if cb_default == 0:
                positive_factors.append({
                    "factor": "Нет истории дефолтов",
                    "value": "Хорошо",
                    "impact": "✅ Снижает риск"
                })
            
            if loan_percent_income < 0.25:
                positive_factors.append({
                    "factor": "Низкая доля дохода на кредит",
                    "value": f"{loan_percent_income*100:.1f}%",
                    "impact": "✅ Снижает риск"
                })
            
            if loan_grade in ['A', 'B']:
                positive_factors.append({
                    "factor": "Высокий кредитный рейтинг заявки",
                    "value": display_label(LOAN_GRADE_LABELS, loan_grade),
                    "impact": "✅ Снижает риск"
                })
            
            if person_emp_length > 5:
                positive_factors.append({
                    "factor": "Стабильный опыт работы",
                    "value": f"{person_emp_length:.1f} лет",
                    "impact": "✅ Снижает риск"
                })
            
            if risk_factors or positive_factors:
                col1, col2 = st.columns(2)
                
                with col1:
                    if risk_factors:
                        st.markdown("#### ⚠️ Факторы риска")
                        for factor in risk_factors:
                            st.warning(f"**{factor['factor']}**: {factor['value']} - {factor['impact']}")
                    else:
                        st.info("Нет явных факторов риска")
                
                with col2:
                    if positive_factors:
                        st.markdown("#### ✅ Позитивные факторы")
                        for factor in positive_factors:
                            st.success(f"**{factor['factor']}**: {factor['value']} - {factor['impact']}")
                    else:
                        st.info("Нет явных позитивных факторов")
            
            # Дополнительная информация
            with st.expander("📋 Технические детали предсказания"):
                prediction_label = "Возврат кредита" if result['prediction'] == 0 else "Дефолт"
                st.json({
                    "Решение модели": result['prediction'],
                    "Класс": prediction_label,
                    "Вероятность дефолта": round(result['probability_default'], 4),
                    "Вероятность возврата": round(result['probability_non_default'], 4),
                    "Уровень риска": result['risk_level']
                })
        
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")
            st.info("Пожалуйста, проверьте введённые данные и попробуйте снова.")

with tab2:
    st.header("📚 Понимание метрик модели")
    
    st.markdown("""
    Эта модель использует несколько важных метрик для оценки качества. 
    Ниже объяснение каждой метрики простым языком для бизнеса.
    """)
    
    st.markdown("---")
    
    # Метрики в виде карточек
    metric_explanations = [
        {
            "name": "Доля верных решений",
            "value": f"{metrics['test_accuracy']*100:.1f}%",
            "description": "Показывает, в скольких случаях из 100 модель сделала правильное предсказание.",
            "business_value": "Из 100 кредитных заявок модель правильно оценила 94. Это означает высокую надёжность системы.",
            "icon": "🎯"
        },
        {
            "name": "Точность предсказанных отказов",
            "value": f"{metrics['test_precision']*100:.1f}%",
            "description": "Когда модель говорит 'отказать', насколько часто она права.",
            "business_value": f"Из всех отказов {metrics['test_precision']*100:.1f}% были правильными. Это значит, что модель редко отказывает хорошим клиентам.",
            "icon": "✅"
        },
        {
            "name": "Полнота поиска дефолтов",
            "value": f"{metrics['test_recall']*100:.1f}%",
            "description": "Из всех реальных дефолтов, сколько модель находит.",
            "business_value": f"Модель находит {metrics['test_recall']*100:.1f}% всех дефолтов. Это критически важно для минимизации потерь банка.",
            "icon": "🔍"
        },
        {
            "name": "ROC-AUC",
            "value": f"{metrics['test_roc_auc']:.3f}",
            "description": "Насколько хорошо модель различает рискованных и безопасных заёмщиков.",
            "business_value": "Модель отлично упорядочивает заёмщиков по уровню риска. Это позволяет принимать более обоснованные решения.",
            "icon": "📊"
        },
        {
            "name": "F1-мера",
            "value": f"{metrics['test_f1']:.3f}",
            "description": "Баланс между точностью и полнотой поиска дефолтов.",
            "business_value": "Хороший баланс между тем, чтобы не пропустить дефолты и не отказать хорошим клиентам.",
            "icon": "⚖️"
        }
    ]
    
    for metric in metric_explanations:
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### {metric['icon']}")
                st.metric("Значение метрики", metric['value'], label_visibility="hidden")
            with col2:
                st.markdown(f"#### {metric['name']}")
                st.markdown(metric['description'])
                st.info(f"**Для бизнеса:** {metric['business_value']}")
            st.markdown("---")

    with st.expander("🧮 Формулы расчета метрик", expanded=True):
        st.markdown("""
        В формулах используются обозначения из матрицы ошибок:

        - **TP** – найденные дефолты: заемщик действительно допустил дефолт, и модель правильно отнесла его к рискованным;
        - **TN** – правильные одобрения: заемщик надежный, и модель правильно отнесла его к надежным;
        - **FP** – ложные отказы: заемщик надежный, но модель ошибочно отнесла его к рискованным;
        - **FN** – пропущенные дефолты: заемщик допустил дефолт, но модель ошибочно отнесла его к надежным.
        """)

        st.latex(r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}")
        st.caption("Доля всех правильных решений модели.")

        st.latex(r"Precision = \frac{TP}{TP + FP}")
        st.caption("Точность предсказанных отказов: насколько часто модель права, когда относит заемщика к рискованным.")

        st.latex(r"Recall = \frac{TP}{TP + FN}")
        st.caption("Полнота поиска дефолтов: какую долю реальных дефолтов модель нашла.")

        st.latex(r"F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}")
        st.caption("Баланс между точностью отказов и полнотой поиска дефолтов.")

        st.latex(r"TPR_k = \frac{TP_k}{TP_k + FN_k}")
        st.latex(r"FPR_k = \frac{FP_k}{FP_k + TN_k}")
        st.latex(r"ROC\text{-}AUC = \sum_{k=1}^{m-1} \frac{TPR_k + TPR_{k+1}}{2} \cdot (FPR_{k+1} - FPR_k)")
        st.caption("ROC-AUC считается как сумма площадей трапеций под ROC-кривой при разных порогах решения.")
    
    # Матрица ошибок
    st.markdown("### 📋 Матрица ошибок")
    
    cm = metrics['confusion_matrix']
    
    st.markdown("""
    Матрица ошибок показывает, сколько раз модель правильно и неправильно 
    предсказала исход кредита:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success(f"**Правильные одобрения**\n\n{cm['tn']}")
        st.caption("Хорошие клиенты, которым одобрили")
    
    with col2:
        st.warning(f"**Ложные отказы**\n\n{cm['fp']}")
        st.caption("Хорошие клиенты, которым отказали (потеря прибыли)")
    
    with col3:
        st.error(f"**Пропущенные дефолты**\n\n{cm['fn']}")
        st.caption("Проблемные клиенты, которым одобрили (КРИТИЧНО!)")
    
    with col4:
        st.success(f"**Найденные дефолты**\n\n{cm['tp']}")
        st.caption("Проблемные клиенты, которым правильно отказали")
    
    st.markdown("---")
    st.markdown("### 💡 Выводы")
    st.info(f"""
    **Сильные стороны модели:**
    - Высокая точность ({metrics['test_accuracy']*100:.1f}%)
    - Редко ошибается при отказе ({metrics['test_precision']*100:.1f}% точности)
    - Хорошо упорядочивает заёмщиков по риску (ROC-AUC = {metrics['test_roc_auc']:.3f})
    
    **Области для улучшения:**
    - Модель находит {metrics['test_recall']*100:.1f}% дефолтов, что означает пропуск примерно 
    {100-metrics['test_recall']*100:.1f}% проблемных заёмщиков
    - Это может быть улучшено настройкой порога принятия решения или использованием 
    дополнительных данных
    """)

with tab3:
    st.header("ℹ️ О системе оценки кредитного риска")
    
    st.markdown("""
    ## 🎯 Что делает эта система?
    
    Эта система использует искусственный интеллект и машинное обучение для оценки 
    вероятности того, что заёмщик вернёт кредит или нет (дефолт).
    
    Модель была обучена на исторических данных тысяч кредитных заявок и научилась 
    находить закономерности между характеристиками заёмщика и вероятностью дефолта.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🤖 Как работает модель?")
    
    st.markdown("""
    1. **Сбор данных:** Вы вводите информацию о заёмщике и кредите
    2. **Обработка:** Система преобразует данные в формат, понятный модели
    3. **Анализ:** Модель сравнивает введённые данные с тысячами исторических случаев
    4. **Предсказание:** Модель рассчитывает вероятность дефолта и делает рекомендацию
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Какие факторы учитывает модель?")
    
    factors_col1, factors_col2 = st.columns(2)
    
    with factors_col1:
        st.markdown("""
        **Личные данные:**
        - Возраст заёмщика
        - Годовой доход
        - Статус жилья
        - Опыт работы
        
        **Кредитная история:**
        - Длина кредитной истории
        - Наличие дефолтов в прошлом
        """)
    
    with factors_col2:
        st.markdown("""
        **Параметры кредита:**
        - Сумма кредита
        - Процентная ставка
        - Доля дохода на кредит
        - Цель кредита
        - Кредитный рейтинг заявки
        """)

    st.markdown("### 🏷️ Что такое кредитный рейтинг заявки?")
    st.info("""
    `Кредитный рейтинг заявки` (`loan_grade`) — это входной признак из исходного
    набора данных. Его обычно присваивает кредитор или скоринговая система до
    финального решения по заявке. Это не субъективная оценка сотрудника и не
    результат работы этой модели.

    Чтобы не заставлять пользователя выбирать этот рейтинг вручную, приложение
    рассчитывает его автоматически при отправке формы. Демо-правило использует
    процентную ставку как основной сигнал и корректирует рейтинг по доле дохода
    на кредит, прошлым дефолтам и стажу работы. В production-сценарии это нужно
    заменить на официальный внутренний рейтинг банка или переобучить модель без
    признака `loan_grade`.
    """)
    with st.expander("Правило расчёта рейтинга в демо-версии"):
        st.markdown(LOAN_GRADE_RULE_DETAILS)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Как интерпретировать результаты?")
    
    st.markdown("""
    **Вероятность дефолта менее 30%** - ✅ Низкий риск
    - Кредит рекомендуется к одобрению
    - Заёмщик демонстрирует стабильные финансовые показатели
    
    **Вероятность дефолта 30-50%** - ⚠️ Умеренный риск
    - Требуется дополнительный анализ
    - Возможно, нужно запросить дополнительные документы
    
    **Вероятность дефолта более 50%** - ❌ Высокий риск
    - Кредит не рекомендуется к одобрению
    - Высокая вероятность потерь
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛡️ Важные замечания")
    
    st.warning("""
    ⚠️ **Эта система является инструментом поддержки принятия решений, 
    а не заменой человеческого анализа.**
    
    - Всегда учитывайте контекст и дополнительные факторы
    - Решение о выдаче кредита должно приниматься с учётом политики банка
    - Модель показывает вероятность, но не гарантирует результат
    - Рекомендуется использовать систему в комплексе с другими методами анализа
    """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Технические характеристики")
    
    tech_info = {
        "Тип модели": display_label(MODEL_TYPE_LABELS, metadata['model_type']),
        "Обучение": "Градиентный бустинг",
        "Доля верных решений": f"{metrics['test_accuracy']*100:.2f}%",
        "ROC-AUC": f"{metrics['test_roc_auc']:.3f}",
        "Дата создания": "2024"
    }
    
    st.json(tech_info)
    
    st.markdown("---")
    
    st.markdown("### 📞 Контакты и поддержка")
    
    st.info("""
    Если у вас возникли вопросы о работе системы или нужна помощь в интерпретации 
    результатов, пожалуйста, обратитесь к команде аналитики данных вашей организации.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Система оценки кредитного риска | На базе машинного обучения
    </div>
    """,
    unsafe_allow_html=True
)
