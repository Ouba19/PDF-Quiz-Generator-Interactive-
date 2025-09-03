import streamlit as st
from PyPDF2 import PdfReader
import random
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# --- Préparations NLTK (si nécessaire) ---
@st.cache_resource
def initialize_nltk():
    try:
        _ = stopwords.words('french')
        _ = sent_tokenize("Test.", language='french')
        _ = word_tokenize("Test", language='french')
    except Exception:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')  # Nouvelle version de punkt
    return set(stopwords.words('french'))

STOP_WORDS = initialize_nltk()

# --- Fonctions utilitaires ---

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extrait le texte d'un fichier PDF téléversé."""
    try:
        reader = PdfReader(uploaded_file)
        texts = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    texts.append(text)
            except Exception as e:
                st.warning(f"Erreur lors de l'extraction d'une page: {e}")
                continue
        
        if not texts:
            return ""
        
        return '\n'.join(texts)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF: {e}")
        return ""


def clean_text(text):
    """Nettoie et normalise le texte."""
    if not text:
        return ""
    
    # Suppression des caractères de contrôle et normalisation des espaces
    text = text.replace('\r', ' ')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Suppression des caractères non imprimables
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text.strip()


def candidate_words_from_sentence(sentence):
    """Extrait les mots candidats d'une phrase pour les questions."""
    try:
        words = word_tokenize(sentence, language='french')
        candidates = []
        
        for word in words:
            # Mot alphabétique, longueur > 3, pas un mot vide, pas trop long
            if (word.isalpha() and 
                len(word) > 3 and 
                len(word) < 15 and
                word.lower() not in STOP_WORDS):
                candidates.append(word)
        
        return candidates
    except Exception:
        # Fallback si NLTK ne fonctionne pas
        words = sentence.split()
        return [w for w in words if w.isalpha() and len(w) > 3 and len(w) < 15]


def generate_better_distractors(answer, word_pool, max_distractors=3):
    """Génère de meilleurs distracteurs pour une réponse donnée."""
    distractors = []
    answer_lower = answer.lower()
    
    # 1. Mots de longueur similaire du pool
    similar_length = [w for w in word_pool 
                     if abs(len(w) - len(answer)) <= 2 
                     and w.lower() != answer_lower]
    
    if similar_length:
        distractors.extend(random.sample(similar_length, 
                                       min(2, len(similar_length))))
    
    # 2. Mots commençant par la même lettre
    same_start = [w for w in word_pool 
                  if w.lower().startswith(answer_lower[0]) 
                  and w.lower() != answer_lower
                  and w not in distractors]
    
    if same_start:
        distractors.extend(random.sample(same_start, 
                                       min(1, len(same_start))))
    
    # 3. Compléter avec des mots aléatoires si nécessaire
    remaining_pool = [w for w in word_pool 
                     if w.lower() != answer_lower 
                     and w not in distractors]
    
    while len(distractors) < max_distractors and remaining_pool:
        distractors.append(random.choice(remaining_pool))
        remaining_pool.remove(distractors[-1])
    
    # 4. Si toujours pas assez, créer des distracteurs artificiels
    while len(distractors) < max_distractors:
        if len(answer) > 4:
            fake = answer[:3] + answer[4:]  # Supprime un caractère
        else:
            fake = answer + "e"  # Ajoute une lettre
        
        if fake not in distractors and fake.lower() != answer_lower:
            distractors.append(fake)
        else:
            break
    
    return distractors[:max_distractors]


def generate_fill_blank_question(sentence, word_pool):
    """Génère une question à trous à partir d'une phrase."""
    candidates = candidate_words_from_sentence(sentence)
    
    if not candidates:
        return None
    
    # Choisir le mot à cacher (préférer les mots plus longs et significatifs)
    answer = max(candidates, key=len) if candidates else random.choice(candidates)
    
    # Créer la question en remplaçant le mot par des blancs
    pattern = r'\b' + re.escape(answer) + r'\b'
    question_text = re.sub(pattern, '_____', sentence, count=1, flags=re.IGNORECASE)
    
    # Vérifier que le remplacement a eu lieu
    if question_text == sentence:
        return None
    
    # Générer les distracteurs
    distractors = generate_better_distractors(answer, word_pool)
    
    if len(distractors) < 2:  # Pas assez de distracteurs
        return None
    
    # Créer les options
    options = distractors + [answer]
    random.shuffle(options)
    
    return {
        'question': question_text,
        'answer': answer,
        'options': options,
        'origin': sentence
    }


def build_questions_from_text(text, max_questions=10):
    """Construit une liste de questions à partir d'un texte."""
    if not text:
        return []
    
    text = clean_text(text)
    
    try:
        sentences = sent_tokenize(text, language='french')
    except Exception:
        # Fallback si NLTK ne fonctionne pas
        sentences = text.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    # Filtrer les phrases appropriées
    good_sentences = []
    for sentence in sentences:
        word_count = len(sentence.split())
        # Phrases ni trop courtes ni trop longues
        if 8 <= word_count <= 35 and len(sentence) > 50:
            good_sentences.append(sentence)
    
    if not good_sentences:
        return []
    
    # Construire le pool de mots pour les distracteurs
    word_pool = []
    for sentence in good_sentences:
        word_pool.extend(candidate_words_from_sentence(sentence))
    
    word_pool = list(set(word_pool))  # Supprimer les doublons
    
    if len(word_pool) < 10:  # Pas assez de vocabulaire
        return []
    
    # Mélanger les phrases et générer les questions
    random.shuffle(good_sentences)
    questions = []
    
    for sentence in good_sentences:
        if len(questions) >= max_questions:
            break
            
        question = generate_fill_blank_question(sentence, word_pool)
        if question:
            questions.append(question)
    
    return questions

# --- Interface Streamlit ---

def main():
    st.set_page_config(
        page_title='Générateur de Quiz PDF', 
        layout='centered',
        initial_sidebar_state='expanded'
    )
    
    st.title('🎯 Générateur de Quiz Interactifs')
    st.markdown("""
    **Créez automatiquement des quiz à partir de vos documents PDF !**
    
    📝 **Comment ça marche :**
    1. Téléversez un PDF contenant du texte
    2. L'application extrait les phrases importantes
    3. Des questions à trous sont générées automatiquement
    4. Testez vos connaissances avec le quiz interactif
    
    > ⚠️ **Note :** Ceci est un prototype. Pour des quiz professionnels, 
    > une révision humaine est recommandée.
    """)
    
    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres")
    num_questions = st.sidebar.slider(
        'Nombre de questions', 
        min_value=3, 
        max_value=25, 
        value=10,
        help="Plus le texte est long, plus vous pouvez générer de questions"
    )
    
    # Zone de téléversement
    uploaded_file = st.file_uploader(
        '📁 Téléversez votre PDF', 
        type=['pdf'],
        help="Assurez-vous que le PDF contient du texte (pas seulement des images)"
    )
    
    if uploaded_file is not None:
        # Afficher les informations du fichier
        st.info(f"📄 Fichier: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Extraction du texte
        with st.spinner('🔄 Extraction du texte en cours...'):
            raw_text = extract_text_from_pdf(uploaded_file)
        
        if not raw_text.strip():
            st.error("""
            ❌ **Impossible d'extraire le texte du PDF**
            
            Causes possibles :
            - Le PDF contient uniquement des images
            - Le PDF est protégé par un mot de passe
            - Le fichier est corrompu
            
            **Solution :** Essayez avec un PDF contenant du texte sélectionnable.
            """)
            return
        
        # Afficher un aperçu du texte
        st.success(f"✅ Texte extrait ({len(raw_text)} caractères)")
        
        with st.expander("👁️ Aperçu du texte extrait"):
            st.text(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
        
        # Génération des questions
        st.markdown("---")
        st.subheader("🎲 Génération du Quiz")
        
        if st.button("🚀 Générer le Quiz", type="primary"):
            with st.spinner('🧠 Génération des questions...'):
                questions = build_questions_from_text(raw_text, max_questions=num_questions)
            
            if not questions:
                st.warning("""
                ⚠️ **Aucune question n'a pu être générée**
                
                Raisons possibles :
                - Le texte est trop court
                - Le vocabulaire est insuffisant
                - Les phrases sont trop courtes ou trop longues
                
                **Suggestion :** Essayez avec un document plus long et structuré.
                """)
                return
            
            # Stocker les questions dans la session
            st.session_state.questions = questions
            st.session_state.quiz_generated = True
        
        # Afficher le quiz si généré
        if hasattr(st.session_state, 'quiz_generated') and st.session_state.quiz_generated:
            st.markdown("---")
            st.subheader(f"📝 Quiz ({len(st.session_state.questions)} questions)")
            
            # Formulaire de quiz
            with st.form("quiz_form"):
                user_answers = []
                
                for i, question in enumerate(st.session_state.questions, 1):
                    st.markdown(f"**Question {i}**")
                    st.write(question['question'])
                    
                    choice = st.radio(
                        f"Choisissez la réponse pour la question {i}:",
                        options=question['options'],
                        key=f'question_{i}',
                        index=None
                    )
                    
                    user_answers.append({
                        'question_num': i,
                        'selected': choice,
                        'correct': question['answer'],
                        'origin': question['origin'],
                        'question_text': question['question']
                    })
                    
                    st.markdown("---")
                
                submitted = st.form_submit_button("📊 Obtenir mes Résultats", type="primary")
                
                if submitted:
                    # Calculer le score
                    answered_questions = [a for a in user_answers if a['selected'] is not None]
                    correct_answers = [a for a in answered_questions if a['selected'] == a['correct']]
                    
                    if not answered_questions:
                        st.warning("⚠️ Veuillez répondre à au moins une question !")
                        return
                    
                    score_percentage = (len(correct_answers) / len(answered_questions)) * 100
                    
                    # Afficher les résultats
                    st.markdown("## 🏆 Résultats du Quiz")
                    
                    # Métriques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score", f"{len(correct_answers)}/{len(answered_questions)}")
                    with col2:
                        st.metric("Pourcentage", f"{score_percentage:.1f}%")
                    with col3:
                        if score_percentage >= 80:
                            st.metric("Niveau", "Excellent 🌟")
                        elif score_percentage >= 60:
                            st.metric("Niveau", "Bien 👍")
                        else:
                            st.metric("Niveau", "À améliorer 📚")
                    
                    # Détail des réponses
                    st.markdown("### 📋 Détail des réponses")
                    
                    for answer in answered_questions:
                        is_correct = answer['selected'] == answer['correct']
                        
                        with st.expander(
                            f"Question {answer['question_num']} - "
                            f"{'✅ Correct' if is_correct else '❌ Incorrect'}"
                        ):
                            st.write(f"**Question :** {answer['question_text']}")
                            st.write(f"**Votre réponse :** {answer['selected']}")
                            st.write(f"**Bonne réponse :** {answer['correct']}")
                            st.write(f"**Phrase d'origine :** {answer['origin']}")
    
    else:
        st.markdown("""
        ### 🚀 Pour commencer :
        
        1. **Cliquez** sur "Parcourir les fichiers" ci-dessus
        2. **Sélectionnez** un fichier PDF contenant du texte
        3. **Ajustez** le nombre de questions souhaité dans la sidebar
        4. **Générez** votre quiz personnalisé !
        
        ### 💡 Conseils pour de meilleurs résultats :
        
        - Utilisez des PDF avec du texte **bien structuré**
        - Les documents de **cours** ou **articles** fonctionnent mieux
        - Plus le texte est **long et varié**, meilleures sont les questions
        - Évitez les PDF avec uniquement des **images** ou **graphiques**
        """)

if __name__ == "__main__":
    main()