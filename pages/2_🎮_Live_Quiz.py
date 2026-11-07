import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="Live Quiz & Scoring — DocuMind",
    page_icon="🎮",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">🎮 Active Knowledge Assessment</div>
    <div class="brand-heading">Gamified Live Quiz</div>
    <div class="brand-tagline">
        Test your understanding of uploaded textbooks. Interactive questions with instant grading, scoring, and book citations.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vector_store.chunks:
    st.info("💡 **Getting Started:** Upload your book in the sidebar, or click **'📑 Load Sample'** to generate a quiz.")
else:
    quiz_col1, quiz_col2 = st.columns([3, 1])
    with quiz_col1:
        quiz_topic = st.text_input("Quiz Topic or Chapter Filter (Optional)", placeholder="e.g. Multithreading, Attention, CPU Scheduling, Virtual Memory, etc.")
    with quiz_col2:
        quiz_q_count = st.selectbox("Number of Questions", options=[3, 4, 5], index=0)

    if st.button("⚡ Generate New Live Quiz", use_container_width=True):
        with st.spinner("Reviewing book content & generating interactive questions..."):
            quiz_data = pipeline.generate_interactive_quiz(topic=quiz_topic, count=quiz_q_count)
            st.session_state.interactive_quiz = quiz_data
            st.session_state.user_quiz_answers = {}
            st.session_state.quiz_score = None
            st.rerun()

    if st.session_state.interactive_quiz:
        st.markdown("---")
        with st.form("interactive_quiz_form"):
            for q_idx, q in enumerate(st.session_state.interactive_quiz):
                st.markdown(f"**Q{q_idx + 1}: {q['question']}** (Page {q.get('page', 1)})")
                selected_opt = st.radio(
                    f"Select answer for Q{q_idx + 1}:",
                    options=q["options"],
                    key=f"user_ans_q_{q_idx}",
                    label_visibility="collapsed"
                )
                st.session_state.user_quiz_answers[q_idx] = selected_opt
                st.write("")

            submitted = st.form_submit_button("🏆 Submit Answers & Get Score", use_container_width=True)

        if submitted:
            correct_count = 0
            total_questions = len(st.session_state.interactive_quiz)
            
            st.markdown("### 📊 Your Results & Scorecard:")
            for q_idx, q in enumerate(st.session_state.interactive_quiz):
                user_choice = st.session_state.user_quiz_answers.get(q_idx)
                correct_choice = q["options"][q["correct_index"]]
                is_correct = (user_choice == correct_choice)
                
                if is_correct:
                    correct_count += 1
                    st.success(f"✅ **Q{q_idx + 1}: Correct!** Selected: *{user_choice}*")
                else:
                    st.error(f"❌ **Q{q_idx + 1}: Incorrect!** Selected: *{user_choice}* | Correct Answer: **{correct_choice}**")
                
                st.info(f"💡 **Book Explanation:** {q['explanation']}")
                st.markdown("---")

            pct = int((correct_count / total_questions) * 100)
            st.markdown(f"## 🎯 Final Score: **{correct_count} / {total_questions} ({pct}%)**")
            if pct >= 70:
                st.balloons()
                st.success("🎉 Outstanding performance! You demonstrated strong mastery of the book text.")
            else:
                st.warning("Good attempt! Review the explanations above to strengthen your concepts.")
