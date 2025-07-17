import streamlit as st
import pandas as pd
import numpy as np
import re

# --- Configuration ---
MODEL_DIR = 'models'
LR_MODEL_PATH = os.path.join('sentiment_lr_model.joblib')
MNB_MODEL_PATH = os.path.join( 'sentiment_mnb_model.joblib')
SVC_MODEL_PATH = os.path.join('sentiment_svc_model.joblib')
TFIDF_VECTORIZER_PATH = os.path.join('tfidf_vectorizer.joblib')
CLASS_NAMES = ['Negative', 'Positive']

# --- NLTK Downloads (Cached) ---
@st.cache_resource
def download_nltk_resources():
    st.write("Checking NLTK resources...")
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            if resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            elif resource in ['stopwords', 'wordnet', 'omw-1.4']:
                 nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                st.info(f"Downloading NLTK resource: '{resource}'...")
                nltk.download(resource, quiet=True)
                st.success(f"Downloaded '{resource}'")
            except (URLError, Exception) as e:
                st.error(f"Failed to download NLTK resource '{resource}': {e}")
                st.error("Please check your internet connection or try running `python -m nltk.downloader all` in your terminal.")
                st.stop()
        except Exception as e:
            st.warning(f"An unexpected error occurred while checking '{resource}': {e}. Attempting download.")
            try:
                st.info(f"Downloading NLTK resource: '{resource}'...")
                nltk.download(resource, quiet=True)
                st.success(f"Downloaded '{resource}'")
            except (URLError, Exception) as e:
                st.error(f"Failed to download NLTK resource '{resource}': {e}")
                st.error("Please check your internet connection or try running `python -m nltk.downloader all` in your terminal.")
                st.stop()
    st.success("All NLTK resources checked/downloaded!")
    return True

download_nltk_resources()


# --- Model and Vectorizer Loading (Cached for performance) ---
@st.cache_resource
def load_all_models_and_vectorizer():
    """Loads all pre-trained models and the TF-IDF vectorizer."""
    models = {}
    model_paths = {
        "Logistic Regression": LR_MODEL_PATH,
        "Multinomial Naive Bayes": MNB_MODEL_PATH,
        "LinearSVC (SVM)": SVC_MODEL_PATH
    }

    for name, path in model_paths.items():
        if not os.path.exists(path):
            st.error(f"Error: {name} model not found at {path}. Please ensure all models are trained and saved.")
            st.stop()
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading {name} model: {e}")
            st.stop()

    if not os.path.exists(TFIDF_VECTORIZER_PATH):
        st.error(f"Error: TF-IDF vectorizer not found at {TFIDF_VECTORIZER_PATH}. Please ensure it is trained and saved.")
        st.stop()
    try:
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    except Exception as e:
        st.error(f"Error loading TF-IDF vectorizer: {e}")
        st.stop()

    return models, vectorizer

with st.spinner("Loading AI models and resources..."):
    all_models, tfidf_vectorizer = load_all_models_and_vectorizer()
st.success("All Models and Vectorizer Loaded!")


# --- Text Preprocessing Functions (Must match training preprocessing) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(filtered_tokens)

def preprocess_review_for_prediction(review_text):
    cleaned_text = clean_text(review_text)
    processed_text = tokenize_and_lemmatize(cleaned_text)
    tfidf_vector = tfidf_vectorizer.transform([processed_text])
    return tfidf_vector

def get_sentiment_prediction(model, preprocessed_vector, confidence_threshold):
    """
    Predicts sentiment and confidence. Handles models that don't have predict_proba.
    """
    prediction = model.predict(preprocessed_vector)[0]
    
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(preprocessed_vector)[0]
        confidence_score = prediction_proba[prediction] * 100
    else: # For LinearSVC, which doesn't have predict_proba
        # For SVC, a common approach is to use decision_function for a "score"
        # but it's not a probability. We'll use a fixed high confidence for simplicity
        # or indicate that probability is not available.
        confidence_score = None # Indicate probability not available
        # You could also use CalibratedClassifierCV if you need probabilities for SVC
        st.warning("Note: LinearSVC does not provide prediction probabilities directly.")
    
    predicted_sentiment = CLASS_NAMES[prediction]

    # Apply confidence threshold
    if confidence_score is not None and confidence_score < confidence_threshold:
        return "Uncertain", confidence_score
    
    return predicted_sentiment, confidence_score

# --- Streamlit UI ---
st.set_page_config(
    page_title="Advanced Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide" # Use wide layout for more space
)

# Custom Styling (expanded)
st.markdown(
    """
    <style>
    .big-font {
        font-size:36px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
    }
    .medium-font {
        font-size:22px !important;
        font-weight: bold;
    }
    .stSpinner > div > div {
        color: #4CAF50;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    .sentiment-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .sentiment-positive { background-color: #28a745; } /* Green */
    .sentiment-negative { background-color: #dc3545; } /* Red */
    .sentiment-uncertain { background-color: #ffc107; color: black !important;} /* Yellow */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="big-font">🎬 Advanced Movie Review Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar for Model Selection and Global Controls ---
with st.sidebar:
    st.header("App Controls")
    selected_model_name = st.selectbox(
        "Choose Model:",
        list(all_models.keys()),
        index=0, # Default to Logistic Regression
        help="Select the machine learning model for sentiment prediction."
    )
    current_model = all_models[selected_model_name]

    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50, max_value=100, value=70, step=1,
        help="Only show sentiment if confidence is above this threshold. Below, it will be marked 'Uncertain'. (Not applicable for LinearSVC)"
    )

    st.markdown("---")
    st.header("About the App")
    st.info("""
    This advanced application allows you to analyze movie review sentiment using
    different machine learning models.
    - **Models:** Logistic Regression, Multinomial Naive Bayes, LinearSVC (SVM).
    - **Preprocessing:** HTML removal, punctuation/number stripping, lowercasing, stop word removal, and lemmatization.
    - **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency) vectors.
    """)
    st.write("---")
    st.write("Developed for an AI/ML Hackathon by Hemanth Kumar")
    st.write(f"App running in local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


# --- Main Content - Using Tabs for UI/UX ---
tab1, tab2 = st.tabs(["✍️ Single Review Analysis", "📁 Batch Review Analysis"])

with tab1:
    st.markdown("## Single Review Analysis")
    st.markdown("Enter a movie review below to get its sentiment prediction.")

    user_review_single = st.text_area(
        "Your Movie Review:",
        "This film was an absolute masterpiece! The performances were breathtaking and the story was incredibly moving. Highly recommend for a truly emotional experience.",
        height=200,
        key="single_review_input" # Unique key for this widget
    )

    col_btn1, col_btn2, _ = st.columns([0.2, 0.2, 0.6])

    analyze_button = col_btn1.button("Analyze Sentiment", key="analyze_single_btn")
    clear_button = col_btn2.button("Clear/Reset", key="clear_single_btn")

    if clear_button:
        st.session_state.single_review_input = "" # Clear text area using session state
        st.experimental_rerun() # Rerun to clear outputs

    if analyze_button and user_review_single.strip():
        with st.spinner(f"Analyzing sentiment with {selected_model_name}..."):
            time.sleep(0.5) # Simulate processing time

            preprocessed_input = preprocess_review_for_prediction(user_review_single)
            predicted_sentiment, confidence_score = get_sentiment_prediction(current_model, preprocessed_input, confidence_threshold)

        st.markdown("### Analysis Results:")
        
        # Display sentiment with custom styling and animations
        if predicted_sentiment == "Positive":
            st.markdown(f'<div class="sentiment-box sentiment-positive">Predicted Sentiment: {predicted_sentiment} 😃</div>', unsafe_allow_html=True)
            st.balloons()
        elif predicted_sentiment == "Negative":
            st.markdown(f'<div class="sentiment-box sentiment-negative">Predicted Sentiment: {predicted_sentiment} 😞</div>', unsafe_allow_html=True)
            st.snow() # Light snow animation for negative (can change)
        else: # Uncertain
            st.markdown(f'<div class="sentiment-box sentiment-uncertain">Predicted Sentiment: {predicted_sentiment} ⚠️</div>', unsafe_allow_html=True)

        # Display confidence if available
        if confidence_score is not None:
            st.metric(label="Confidence", value=f"{confidence_score:.2f}%")
            st.progress(int(confidence_score) / 100.0)
        else:
            st.info("Confidence score not available for this model (e.g., LinearSVC).")

        with st.expander("Show detailed probabilities (where available)"):
            if hasattr(current_model, 'predict_proba'):
                prediction_proba = current_model.predict_proba(preprocessed_input)[0]
                prob_df = pd.DataFrame({
                    "Sentiment": CLASS_NAMES,
                    "Probability": prediction_proba
                })
                prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
            else:
                st.write("Probability estimates are not available for the selected model.")

    elif analyze_button and not user_review_single.strip():
        st.warning("Please enter some text to analyze.")


with tab2:
    st.markdown("## Batch Review Analysis")
    st.markdown("Upload a text file (.txt) with one review per line or a CSV file (.csv) with a 'review' column to analyze multiple reviews at once.")

    uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv"])

    if uploaded_file is not None:
        reviews_to_analyze = []
        file_type = uploaded_file.type

        if file_type == "text/plain":
            stringio = uploaded_file.getvalue().decode("utf-8")
            reviews_to_analyze = stringio.splitlines()
            reviews_to_analyze = [line.strip() for line in reviews_to_analyze if line.strip()]
        elif file_type == "text/csv":
            df_uploaded = pd.read_csv(uploaded_file)
            if 'review' in df_uploaded.columns:
                reviews_to_analyze = df_uploaded['review'].tolist()
            else:
                st.error("CSV file must contain a 'review' column.")
                reviews_to_analyze = []

        if reviews_to_analyze:
            st.info(f"Found {len(reviews_to_analyze)} reviews to analyze using {selected_model_name}.")
            
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, review in enumerate(reviews_to_analyze):
                with st.spinner(f"Processing review {i+1}/{len(reviews_to_analyze)}..."):
                    preprocessed_vector = preprocess_review_for_prediction(review)
                    predicted_sentiment, confidence_score = get_sentiment_prediction(current_model, preprocessed_vector, confidence_threshold)
                    
                    batch_results.append({
                        "Review": review[:100] + "..." if len(review) > 100 else review, # Truncate for display
                        "Predicted Sentiment": predicted_sentiment,
                        "Confidence (%)": f"{confidence_score:.2f}" if confidence_score is not None else "N/A"
                    })
                
                progress_bar.progress((i + 1) / len(reviews_to_analyze))
                status_text.text(f"Processed {i+1} of {len(reviews_to_analyze)} reviews.")
            
            st.success("Batch analysis complete!")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            csv_output = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name="sentiment_batch_results.csv",
                mime="text/csv",
            )
        else:
            if uploaded_file is not None: # Only show error if file was uploaded but no reviews found
                st.warning("No valid reviews found in the uploaded file.")

st.markdown("---")
st.markdown("💡 This application demonstrates sentiment analysis using machine learning. For critical applications, always verify model predictions.")
