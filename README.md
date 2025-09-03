# PDF Quiz Generator (Interactive)

**Description:**
This is an interactive quiz generator that extracts text from PDF documents and creates multiple-choice and fill-in-the-blank questions. Users can upload PDF files containing text (lecture notes, articles, or reports), and the app automatically generates quizzes with 4 options per question (1 correct answer + 3 distractors). The app is built as a prototype with heuristic methods, and more advanced NLP models can be integrated for better question quality.

**Features:**
- Upload PDF files and extract text.
- Automatically generate fill-in-the-blank questions.
- Create multiple-choice questions (MCQ) with distractors.
- Interactive interface to answer questions and get immediate scores.
- View original sentences for context.

**Installation:**
1. Clone or download this repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate   # Linux/Mac
   ```
3. Install required packages:
   ```bash
   pip install streamlit PyPDF2 nltk
   ```
4. Download NLTK resources (if needed):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

**Usage:**
```bash
streamlit run Quiz_ISPITSG.py
```
Upload a PDF and adjust the number of questions using the sidebar slider.

**Notes:**
- Works best with text-based PDFs, not scanned images.
- Prototype uses simple heuristics; consider integrating spaCy or Transformers for advanced NLP-based question generation.

**Future Improvements:**
- Use NLP models (spaCy, BERT, T5) to improve question quality and distractors.
- Add OCR support for scanned PDFs.
- Generate different types of questions (True/False, matching, factual MCQs).
- Customize questions for specific fields, such as medical or scientific content.

**Author:** Ouba19

