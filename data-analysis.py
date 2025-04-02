import streamlit as st
import pandas as pd
import numpy as np
import collections
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import traceback
import re

# NEW Imports for Intent Classification
from transformers import pipeline
import torch # Or import tensorflow if using TF backend for transformers

# ------------------------------------
# Helper Functions
# ------------------------------------

# --- NLTK Setup ---
@st.cache_data # Cache the download check result
def ensure_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
ensure_nltk_data() # Run the check/download on script start

# Load stopwords - Error handling added
try:
    nltk_stop_words = set(stopwords.words('english'))
    nltk_stop_words.update(['a', 'an', 'the', 'of', 'in', 'on', 'at', 'for', 'to', 'is', 'and', 'com', 'www']) # Added common web noise
except Exception as e:
    st.error(f"Error loading NLTK stopwords: {e}. Using a basic fallback list.")
    # Fallback list (minimal)
    nltk_stop_words = set(['and', 'the', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that'])

# --- Topic Label Generation (Revised for Bi-grams) ---
def generate_topic_label(queries_in_topic):
    """Generates a simple topic label from the most common BI-GRAMS in a list of queries."""
    bigram_list = []
    for query in queries_in_topic:
        if isinstance(query, str):
             tokens = [word for word in re.findall(r'\b\w+\b', query.lower()) if len(word) > 1] # Tokenize and basic filter
             # Create bi-grams
             query_bigrams = list(nltk.ngrams(tokens, 2))
             # Filter bi-grams containing stopwords (either word)
             filtered_bigrams = [
                 bg for bg in query_bigrams
                 if bg[0] not in nltk_stop_words and bg[1] not in nltk_stop_words
             ]
             # Add the string representation of valid bi-grams
             bigram_list.extend([" ".join(bg) for bg in filtered_bigrams])

    if bigram_list:
        freq = collections.Counter(bigram_list)
        # Get top 1-2 most common bi-grams
        common = freq.most_common(2)
        label = ", ".join([bg for bg, count in common])
        return label.title() if label else "Topic Cluster" # Capitalize words
    elif queries_in_topic: # Fallback if no valid bigrams found
         # Use single words as fallback for label
         words = []
         for query in queries_in_topic:
             if isinstance(query, str): words.extend([w for w in re.findall(r'\b\w+\b', query.lower()) if w not in nltk_stop_words and len(w)>1])
         if words: return collections.Counter(words).most_common(1)[0][0].capitalize()
         else: return "Unnamed Topic" # Final fallback
    else:
        return "Unnamed Topic"

# --- Intent Classification Setup (Zero-Shot) ---
@st.cache_resource # Cache the pipeline object
def load_intent_classifier():
    # You can choose different models optimized for zero-shot
    # facebook/bart-large-mnli is a common starting point
    # MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli often performs well
    model_name = "facebook/bart-large-mnli"
    st.info(f"Loading Zero-Shot classification model ({model_name})... (This may take a moment on first run)")
    try:
        classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1) # Use GPU if available
        st.success("Intent classification model loaded.")
        return classifier
    except Exception as e:
        st.error(f"Error loading intent classification model: {e}")
        st.error(traceback.format_exc())
        st.warning("Intent classification will not be available.")
        return None

# Function to classify intent using the zero-shot pipeline
def classify_intent_zero_shot(query, classifier, candidate_labels):
    if not classifier or not isinstance(query, str) or not query.strip():
        return "Unknown" # Return Unknown if classifier failed or query invalid
    try:
        # Increase max_length if queries are long, but be mindful of model limits
        result = classifier(query, candidate_labels, multi_label=False) # Set multi_label=False if only one intent expected
        # The label with the highest score is the prediction
        return result['labels'][0]
    except Exception as e:
        # st.warning(f"Error classifying intent for query '{query}': {e}") # Optional: more verbose error
        return "Unknown" # Fallback on error

# Define the candidate labels for zero-shot classification
INTENT_LABELS = ["Informational", "Navigational", "Commercial", "Transactional"]

# --- Other Helper Functions ---
def parse_ctr_original(ctr):
    try:
        if isinstance(ctr, str) and "%" in ctr: return float(ctr.replace("%", "").replace(',', ''))
        else: return pd.to_numeric(ctr, errors='coerce')
    except Exception: return np.nan

def calculate_pct_change(val_after, val_before):
    val_after = pd.to_numeric(val_after, errors='coerce')
    val_before = pd.to_numeric(val_before, errors='coerce')
    if pd.isna(val_before) or pd.isna(val_after) or val_before == 0: return np.nan
    return ((val_after - val_before) / val_before) * 100

# ------------------------------------
# GSC Analyzer Tool Function
# ------------------------------------
def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        Compare GSC query data from two periods. Performs **bi-gram topic modeling** (LDA), classifies **search intent** (using a zero-shot ML model), calculates YoY changes, aggregates metrics, and visualizes results.
        *Requires 'Top queries' and 'Position' columns. 'Clicks', 'Impressions', 'CTR' recommended.*
        *Compares only queries present in **both** periods (`inner` merge).*
        *Zero-shot intent classification performance may vary.*
        """
    )

    # Load the intent classifier (cached)
    intent_classifier = load_intent_classifier()

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # --- Initial Data Load and Validation ---
            status_text.text("Reading CSV files...")
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            progress_bar.progress(10); status_text.text("Validating columns...")
            required_cols = {"Top queries", "Position"}
            if not required_cols.issubset(df_before.columns) or not required_cols.issubset(df_after.columns):
                 missing_before = required_cols - set(df_before.columns); missing_after = required_cols - set(df_after.columns)
                 st.error(f"Missing required columns. Before: {missing_before or 'None'}. After: {missing_after or 'None'}.")
                 return
            progress_bar.progress(15); status_text.text("Calculating dashboard summary...")

            # --- Dashboard Summary ---
            # ...(Dashboard code remains the same)...
            st.markdown("## Dashboard Summary")
            df_before_renamed = df_before.rename(columns={"Top queries": "Query", "Position": "Average Position"})
            df_after_renamed = df_after.rename(columns={"Top queries": "Query", "Position": "Average Position"})
            cols = st.columns(4)
            if "Clicks" in df_before_renamed.columns and "Clicks" in df_after_renamed.columns:
                total_clicks_before = pd.to_numeric(df_before_renamed["Clicks"], errors='coerce').fillna(0).sum(); total_clicks_after = pd.to_numeric(df_after_renamed["Clicks"], errors='coerce').fillna(0).sum()
                overall_clicks_change = total_clicks_after - total_clicks_before; overall_clicks_change_pct = (overall_clicks_change / total_clicks_before * 100) if total_clicks_before != 0 else 0
                cols[0].metric(label="Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
            else: cols[0].metric(label="Clicks Change", value="N/A")
            if "Impressions" in df_before_renamed.columns and "Impressions" in df_after_renamed.columns:
                total_impressions_before = pd.to_numeric(df_before_renamed["Impressions"], errors='coerce').fillna(0).sum(); total_impressions_after = pd.to_numeric(df_after_renamed["Impressions"], errors='coerce').fillna(0).sum()
                overall_impressions_change = total_impressions_after - total_impressions_before; overall_impressions_change_pct = (overall_impressions_change / total_impressions_before * 100) if total_impressions_before != 0 else 0
                cols[1].metric(label="Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
            else: cols[1].metric(label="Impressions Change", value="N/A")
            overall_avg_position_before = pd.to_numeric(df_before_renamed["Average Position"], errors='coerce').mean(); overall_avg_position_after = pd.to_numeric(df_after_renamed["Average Position"], errors='coerce').mean()
            overall_position_change = overall_avg_position_before - overall_avg_position_after; overall_position_change_pct = (overall_position_change / overall_avg_position_before * 100) if pd.notna(overall_avg_position_before) and overall_avg_position_before != 0 else 0
            cols[2].metric(label="Avg. Position Change", value=f"{overall_position_change:.1f}", delta=f"{overall_position_change_pct:.1f}%")
            if "CTR" in df_before_renamed.columns and "CTR" in df_after_renamed.columns:
                df_before_renamed["CTR_parsed"] = df_before_renamed["CTR"].apply(parse_ctr_original); df_after_renamed["CTR_parsed"] = df_after_renamed["CTR"].apply(parse_ctr_original)
                overall_ctr_before = df_before_renamed["CTR_parsed"].mean(); overall_ctr_after = df_after_renamed["CTR_parsed"].mean()
                overall_ctr_change = overall_ctr_after - overall_ctr_before; overall_ctr_change_pct = (overall_ctr_change / overall_ctr_before * 100) if pd.notna(overall_ctr_before) and overall_ctr_before != 0 else 0
                cols[3].metric(label="CTR Change", value=f"{overall_ctr_change:.2f}", delta=f"{overall_ctr_change_pct:.1f}%")
            else: cols[3].metric(label="CTR Change", value="N/A")
            progress_bar.progress(30)

            # --- Merge Data & YoY Calculation ---
            status_text.text("Merging datasets & calculating YoY...")
            merged_df = pd.merge(df_before, df_after, on="Top queries", suffixes=("_before", "_after"), how="inner")
            progress_bar.progress(35)
            if merged_df.empty: st.warning("No common queries found."); status_text.text("Stopped: No common queries."); return

            merged_df.rename(columns={"Top queries": "Query", "Position_before": "Average Position_before", "Position_after": "Average Position_after"}, inplace=True)
            # ... (YOY calculation code remains the same) ...
            merged_df["Average Position_before"] = pd.to_numeric(merged_df["Average Position_before"], errors='coerce'); merged_df["Average Position_after"] = pd.to_numeric(merged_df["Average Position_after"], errors='coerce')
            merged_df["Position_YOY"] = merged_df["Average Position_before"] - merged_df["Average Position_after"]
            click_cols_present = "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns; imp_cols_present = "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns; ctr_cols_present = "CTR_before" in merged_df.columns and "CTR_after" in merged_df.columns
            if click_cols_present: merged_df["Clicks_before_num"] = pd.to_numeric(merged_df["Clicks_before"], errors='coerce').fillna(0); merged_df["Clicks_after_num"] = pd.to_numeric(merged_df["Clicks_after"], errors='coerce').fillna(0); merged_df["Clicks_YOY"] = merged_df["Clicks_after_num"] - merged_df["Clicks_before_num"]
            else: merged_df["Clicks_YOY"], merged_df["Clicks_before_num"], merged_df["Clicks_after_num"] = np.nan, np.nan, np.nan
            if imp_cols_present: merged_df["Impressions_before_num"] = pd.to_numeric(merged_df["Impressions_before"], errors='coerce').fillna(0); merged_df["Impressions_after_num"] = pd.to_numeric(merged_df["Impressions_after"], errors='coerce').fillna(0); merged_df["Impressions_YOY"] = merged_df["Impressions_after_num"] - merged_df["Impressions_before_num"]
            else: merged_df["Impressions_YOY"], merged_df["Impressions_before_num"], merged_df["Impressions_after_num"] = np.nan, np.nan, np.nan
            if ctr_cols_present: merged_df["CTR_parsed_before"] = merged_df["CTR_before"].apply(parse_ctr_original); merged_df["CTR_parsed_after"] = merged_df["CTR_after"].apply(parse_ctr_original); merged_df["CTR_YOY"] = merged_df["CTR_parsed_after"] - merged_df["CTR_parsed_before"]
            else: merged_df["CTR_YOY"], merged_df["CTR_parsed_before"], merged_df["CTR_parsed_after"] = np.nan, np.nan, np.nan
            merged_df["Position_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Average Position_after"], row["Average Position_before"]), axis=1)
            if click_cols_present: merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Clicks_after_num"], row["Clicks_before_num"]), axis=1)
            else: merged_df["Clicks_YOY_pct"] = np.nan
            if imp_cols_present: merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Impressions_after_num"], row["Impressions_before_num"]), axis=1)
            else: merged_df["Impressions_YOY_pct"] = np.nan
            if ctr_cols_present: merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["CTR_parsed_after"], row["CTR_parsed_before"]), axis=1)
            else: merged_df["CTR_YOY_pct"] = np.nan


            # --- Intent Classification (Zero-Shot) ---
            status_text.text("Classifying search intent (Zero-Shot)...")
            if intent_classifier:
                 # Apply in batches for potential speedup (optional, depends on data size)
                 # For simplicity, apply row by row first
                 merged_df['Intent'] = merged_df['Query'].apply(
                     lambda q: classify_intent_zero_shot(q, intent_classifier, INTENT_LABELS)
                 )
            else:
                 merged_df['Intent'] = "Unknown" # Fallback if classifier failed to load
                 st.warning("Intent classification skipped as model failed to load.")

            # --- Display Intent Distribution ---
            st.markdown("### Search Intent Classification")
            st.markdown("Distribution of classified intents (using Zero-Shot Model):")
            intent_distribution = merged_df['Intent'].value_counts(normalize=True, dropna=False) * 100 # Include Unknown/NaNs if any
            st.dataframe(intent_distribution.map("{:.1f}%".format).reset_index().rename(columns={'proportion': 'Percentage', 'Intent':'Intent Type'}))
            # Show some examples classified as 'Unknown' by the model (or if classifier failed)
            unknown_queries = merged_df[merged_df['Intent'] == 'Unknown']
            if not unknown_queries.empty:
                 st.write("Sample Queries Classified as 'Unknown':")
                 st.dataframe(unknown_queries[['Query']].sample(min(5, unknown_queries.shape[0]), random_state=1), hide_index=True)
            progress_bar.progress(45)


            # --- Topic Modeling (LDA with Bi-grams) ---
            status_text.text("Performing bi-gram topic modeling (LDA)...")
            st.markdown("### Topic Classification (Bi-grams)")
            n_topics_gsc_lda = st.slider("Select number of topics:", min_value=2, max_value=15, value=5, key="lda_topics_gsc")
            queries = merged_df["Query"].astype(str).tolist()
            merged_df["Query_Topic"] = "Unclassified"; merged_df["Query_Topic_Label"] = -1 # Defaults

            try:
                # --- MODIFIED: Use ngram_range=(2, 2) and adjust min_df ---
                vectorizer_queries_lda = CountVectorizer(
                    stop_words='english', # Use scikit-learn's built-in list
                    ngram_range=(2, 2),   # Use bi-grams ONLY
                    min_df=2,             # Require bi-gram to appear at least twice
                    token_pattern=r'\b[a-zA-Z]{2,}\b' # Keep words >= 2 letters
                )
                query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                if query_matrix_lda.shape[1] > 0: # Check vocabulary size
                    actual_n_topics = min(n_topics_gsc_lda, query_matrix_lda.shape[0])
                    if actual_n_topics < n_topics_gsc_lda: st.warning(f"Reduced topics to {actual_n_topics}.")
                    if actual_n_topics >= 2:
                        lda_queries_model = LatentDirichletAllocation(n_components=actual_n_topics, random_state=42, max_iter=15)
                        lda_queries_model.fit(query_matrix_lda)
                        query_topic_labels = lda_queries_model.transform(query_matrix_lda).argmax(axis=1)
                        merged_df["Query_Topic_Label"] = query_topic_labels
                        topic_labels_desc_queries = {}
                        for topic_idx in range(actual_n_topics):
                             topic_queries_lda = merged_df[merged_df["Query_Topic_Label"] == topic_idx]["Query"].tolist()
                             # Use the generate_topic_label function (now expecting bi-grams)
                             topic_labels_desc_queries[topic_idx] = generate_topic_label(topic_queries_lda)
                        merged_df["Query_Topic"] = merged_df["Query_Topic_Label"].apply(lambda x: topic_labels_desc_queries.get(x, f"Topic {x+1}"))

                        st.write("Identified Query Topics (Top Bi-grams):")
                        for topic_idx, topic_comp in enumerate(lda_queries_model.components_):
                            # Get top bi-grams for the topic
                            top_feature_indices = topic_comp.argsort()[-10:][::-1]
                            topic_bigrams = [feature_names_queries_lda[i] for i in top_feature_indices]
                            desc_label = topic_labels_desc_queries.get(topic_idx, f"Topic {topic_idx+1}")
                            st.write(f"**{desc_label}:** {', '.join(topic_bigrams)}")
                    else: st.warning("Not enough documents/topics for LDA.")
                else: st.warning("LDA Warning: No bi-gram features found after filtering. Try adjusting CountVectorizer settings (min_df).")

            except Exception as lda_error: st.error(f"Error during LDA: {lda_error}"); st.error(traceback.format_exc())
            progress_bar.progress(55)

            # --- Display Merged Data Table ---
            st.markdown("### Detailed Query Data by Topic & Intent")
            # Prepare columns for display (including Intent and Topic)
            base_cols = ["Query", "Intent", "Query_Topic", "Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"]
            if click_cols_present: base_cols += ["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"]
            if imp_cols_present: base_cols += ["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"]
            if ctr_cols_present:
                 if "CTR_parsed_before" in merged_df: base_cols += ["CTR_parsed_before", "CTR_parsed_after", "CTR_YOY", "CTR_YOY_pct"]
                 else: base_cols += ["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"]
            final_base_cols = [col for col in base_cols if col in merged_df.columns] # Ensure columns exist
            # Add Topic and Intent if they weren't in final_base_cols yet
            if "Query_Topic" not in final_base_cols: final_base_cols.insert(1, "Query_Topic")
            if "Intent" not in final_base_cols: final_base_cols.insert(1, "Intent")
            merged_df_display_ready = merged_df[final_base_cols].copy()

            # Define Formatting (remains the same)
            format_dict_merged = {} # Define formats...
            float_format = "{:.1f}"; int_format = "{:,.0f}"; pct_format = "{:.2f}%"
            if "Average Position_before" in merged_df_display_ready.columns: format_dict_merged["Average Position_before"] = float_format
            if "Average Position_after" in merged_df_display_ready.columns: format_dict_merged["Average Position_after"] = float_format
            if "Position_YOY" in merged_df_display_ready.columns: format_dict_merged["Position_YOY"] = float_format
            if "Clicks_before" in merged_df_display_ready.columns: format_dict_merged["Clicks_before"] = int_format
            if "Clicks_after" in merged_df_display_ready.columns: format_dict_merged["Clicks_after"] = int_format
            if "Clicks_YOY" in merged_df_display_ready.columns: format_dict_merged["Clicks_YOY"] = int_format
            if "Impressions_before" in merged_df_display_ready.columns: format_dict_merged["Impressions_before"] = int_format
            if "Impressions_after" in merged_df_display_ready.columns: format_dict_merged["Impressions_after"] = int_format
            if "Impressions_YOY" in merged_df_display_ready.columns: format_dict_merged["Impressions_YOY"] = int_format
            ctr_before_col_fmt = "CTR_parsed_before" if "CTR_parsed_before" in merged_df_display_ready.columns else "CTR_before"
            ctr_after_col_fmt = "CTR_parsed_after" if "CTR_parsed_after" in merged_df_display_ready.columns else "CTR_after"
            if ctr_before_col_fmt in merged_df_display_ready.columns: format_dict_merged[ctr_before_col_fmt] = pct_format
            if ctr_after_col_fmt in merged_df_display_ready.columns: format_dict_merged[ctr_after_col_fmt] = pct_format
            if "CTR_YOY" in merged_df_display_ready.columns: format_dict_merged["CTR_YOY"] = "{:,.2f}pp"
            if "Position_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Position_YOY_pct"] = pct_format
            if "Clicks_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Clicks_YOY_pct"] = pct_format
            if "Impressions_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Impressions_YOY_pct"] = pct_format
            if "CTR_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["CTR_YOY_pct"] = pct_format

            st.dataframe(merged_df_display_ready.style.format(format_dict_merged, na_rep="N/A"), use_container_width=True)

            # --- Aggregation by Topic ---
            # ... (Aggregation by Topic code remains the same) ...
            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")
            aggregated_display_final = pd.DataFrame()
            if "Query_Topic" in merged_df.columns and merged_df["Query_Topic"].nunique() > 0 and not merged_df["Query_Topic"].isnull().all():
                agg_dict = {} # Define aggs...
                if "Average Position_before" in merged_df.columns: agg_dict["Average Position_before"] = "mean"
                if "Average Position_after" in merged_df.columns: agg_dict["Average Position_after"] = "mean"
                if "Position_YOY" in merged_df.columns: agg_dict["Position_YOY"] = "mean"
                if "Clicks_before_num" in merged_df.columns: agg_dict["Clicks_before_num"] = "sum"
                if "Clicks_after_num" in merged_df.columns: agg_dict["Clicks_after_num"] = "sum"
                if "Clicks_YOY" in merged_df.columns: agg_dict["Clicks_YOY"] = "sum"
                if "Impressions_before_num" in merged_df.columns: agg_dict["Impressions_before_num"] = "sum"
                if "Impressions_after_num" in merged_df.columns: agg_dict["Impressions_after_num"] = "sum"
                if "Impressions_YOY" in merged_df.columns: agg_dict["Impressions_YOY"] = "sum"
                if "CTR_parsed_before" in merged_df.columns: agg_dict["CTR_parsed_before"] = "mean"
                if "CTR_parsed_after" in merged_df.columns: agg_dict["CTR_parsed_after"] = "mean"
                if "CTR_YOY" in merged_df.columns: agg_dict["CTR_YOY"] = "mean"

                aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index()
                aggregated.rename(columns={"Query_Topic": "Topic", "Clicks_before_num": "Clicks_before", "Clicks_after_num": "Clicks_after", "Impressions_before_num": "Impressions_before", "Impressions_after_num": "Impressions_after", "CTR_parsed_before": "CTR_before", "CTR_parsed_after": "CTR_after"}, inplace=True)
                aggregated["Position_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row.get("Average Position_after"), row.get("Average Position_before")), axis=1)
                if click_cols_present: aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row.get("Clicks_after"), row.get("Clicks_before")), axis=1)
                else: aggregated["Clicks_YOY_pct"] = np.nan
                if imp_cols_present: aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row.get("Impressions_after"), row.get("Impressions_before")), axis=1)
                else: aggregated["Impressions_YOY_pct"] = np.nan
                if ctr_cols_present: aggregated["CTR_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row.get("CTR_after"), row.get("CTR_before")), axis=1)
                else: aggregated["CTR_YOY_pct"] = np.nan
                progress_bar.progress(75)
                new_order = ["Topic"] # Define order...
                if "Average Position_before" in aggregated.columns: new_order.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
                if "Clicks_before" in aggregated.columns: new_order.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
                if "Impressions_before" in aggregated.columns: new_order.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
                if "CTR_before" in aggregated.columns: new_order.extend(["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"])
                final_agg_order = [col for col in new_order if col in aggregated.columns]
                aggregated_display_final = aggregated[final_agg_order].copy()
                format_dict_agg = {} # Define formats...
                if "Average Position_before" in aggregated_display_final.columns: format_dict_agg["Average Position_before"] = float_format
                if "Average Position_after" in aggregated_display_final.columns: format_dict_agg["Average Position_after"] = float_format
                if "Position_YOY" in aggregated_display_final.columns: format_dict_agg["Position_YOY"] = float_format
                if "Clicks_before" in aggregated_display_final.columns: format_dict_agg["Clicks_before"] = int_format
                if "Clicks_after" in aggregated_display_final.columns: format_dict_agg["Clicks_after"] = int_format
                if "Clicks_YOY" in aggregated_display_final.columns: format_dict_agg["Clicks_YOY"] = int_format
                if "Impressions_before" in aggregated_display_final.columns: format_dict_agg["Impressions_before"] = int_format
                if "Impressions_after" in aggregated_display_final.columns: format_dict_agg["Impressions_after"] = int_format
                if "Impressions_YOY" in aggregated_display_final.columns: format_dict_agg["Impressions_YOY"] = int_format
                if "CTR_before" in aggregated_display_final.columns: format_dict_agg["CTR_before"] = pct_format
                if "CTR_after" in aggregated_display_final.columns: format_dict_agg["CTR_after"] = pct_format
                if "CTR_YOY" in aggregated_display_final.columns: format_dict_agg["CTR_YOY"] = "{:,.2f}pp"
                if "Position_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Position_YOY_pct"] = pct_format
                if "Clicks_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Clicks_YOY_pct"] = pct_format
                if "Impressions_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Impressions_YOY_pct"] = pct_format
                if "CTR_YOY_pct" in aggregated_display_final.columns: format_dict_agg["CTR_YOY_pct"] = pct_format

                display_count = st.number_input("Topics to display:", min_value=1, value=min(10, aggregated_display_final.shape[0]), max_value=aggregated_display_final.shape[0], step=1 )
                st.dataframe(aggregated_display_final.head(display_count).style.format(format_dict_agg, na_rep="N/A"), use_container_width=True)
            else: st.warning("Could not aggregate by topic.")
            progress_bar.progress(80)


            # --- Aggregation by Intent ---
            # ... (Aggregation by Intent code remains the same) ...
            status_text.text("Aggregating metrics by search intent...")
            st.markdown("### Aggregated Metrics by Search Intent")
            can_agg_intent = 'Intent' in merged_df.columns and click_cols_present and imp_cols_present
            intent_agg_display = pd.DataFrame() # Initialize
            if not can_agg_intent:
                st.warning("Cannot aggregate by intent: Clicks/Impressions cols missing or Intent classification failed.")
            else:
                intent_agg = merged_df.groupby('Intent').agg( Clicks_Before=('Clicks_before_num', 'sum'), Clicks_After=('Clicks_after_num', 'sum'), Impressions_Before=('Impressions_before_num', 'sum'), Impressions_After=('Impressions_after_num', 'sum'), ).reset_index()
                intent_agg['Clicks YoY %'] = intent_agg.apply(lambda row: calculate_pct_change(row['Clicks_After'], row['Clicks_Before']), axis=1)
                intent_agg['Impressions YoY %'] = intent_agg.apply(lambda row: calculate_pct_change(row['Impressions_After'], row['Impressions_Before']), axis=1)
                intent_agg_display = intent_agg[['Intent', 'Clicks_Before', 'Clicks_After', 'Clicks YoY %', 'Impressions_Before', 'Impressions_After', 'Impressions YoY %']]
                format_dict_intent = { "Clicks_Before": int_format, "Clicks_After": int_format, "Clicks YoY %": pct_format, "Impressions_Before": int_format, "Impressions_After": int_format, "Impressions YoY %": pct_format }
                st.dataframe(intent_agg_display.style.format(format_dict_intent, na_rep="N/A"), use_container_width=True)
            progress_bar.progress(85)


            # --- Visualizations ---
            status_text.text("Generating visualizations...")
            st.markdown("---")
            st.markdown("### Visualizations")

            # --- Topic Visualizations ---
            # ... (Topic viz code remains the same) ...
            st.markdown("#### Performance by Topic")
            if not aggregated_display_final.empty:
                available_topics = aggregated_display_final["Topic"].unique().tolist(); available_topics = [t for t in available_topics if pd.notna(t)]
                selected_topics = st.multiselect("Select topics:", options=available_topics, default=available_topics, key="topic_viz_select")
                aggregated_filtered = aggregated_display_final[aggregated_display_final['Topic'].isin(selected_topics)]
                st.markdown("##### Topic YoY % Change")
                vis_data_pct = [] # Collect data...
                for idx, row in aggregated_filtered.iterrows():
                    topic = row["Topic"]
                    if "Position_YOY_pct" in row and pd.notna(row["Position_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Avg. Position", "YOY % Change": row["Position_YOY_pct"]})
                    if "Clicks_YOY_pct" in row and pd.notna(row["Clicks_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Clicks", "YOY % Change": row["Clicks_YOY_pct"]})
                    if "Impressions_YOY_pct" in row and pd.notna(row["Impressions_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Impressions", "YOY % Change": row["Impressions_YOY_pct"]})
                    if "CTR_YOY_pct" in row and pd.notna(row["CTR_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "CTR", "YOY % Change": row["CTR_YOY_pct"]})
                if vis_data_pct: # Plot if data exists...
                    vis_df_pct = pd.DataFrame(vis_data_pct)
                    try:
                        fig_pct = px.bar(vis_df_pct, x="Topic", y="YOY % Change", color="Metric", barmode="group", title="YoY % Change by Topic", labels={"YOY % Change": "YoY Change (%)"}, text_auto='.1f')
                        fig_pct.update_traces(textposition='outside'); st.plotly_chart(fig_pct, use_container_width=True)
                    except Exception as e: st.error(f"Error plotting Topic YoY %: {e}")
                else: st.warning("No YoY % data for selected topics.")
                st.markdown("##### Topic Raw Clicks (Before vs After)")
                if "Clicks_before" in aggregated_filtered.columns and "Clicks_after" in aggregated_filtered.columns: # Plot if cols exist...
                     try:
                         clicks_df = aggregated_filtered[["Topic", "Clicks_before", "Clicks_after"]]; clicks_melted = clicks_df.melt(id_vars="Topic", value_vars=["Clicks_before", "Clicks_after"], var_name="Period", value_name="Clicks")
                         clicks_melted["Period"] = clicks_melted["Period"].str.replace("Clicks_", "")
                         fig_clicks = px.bar(clicks_melted, x="Topic", y="Clicks", color="Period", barmode="group", title="Total Clicks per Topic", labels={"Clicks": "Total Clicks"}, text_auto=True)
                         fig_clicks.update_traces(textposition='outside'); st.plotly_chart(fig_clicks, use_container_width=True)
                     except Exception as e: st.error(f"Error plotting Topic Clicks: {e}")
                else: st.warning("Cannot plot Topic Raw Clicks.")
                st.markdown("##### Topic Raw Impressions (Before vs After)")
                if "Impressions_before" in aggregated_filtered.columns and "Impressions_after" in aggregated_filtered.columns: # Plot if cols exist...
                     try:
                         impressions_df = aggregated_filtered[["Topic", "Impressions_before", "Impressions_after"]]; impressions_melted = impressions_df.melt(id_vars="Topic", value_vars=["Impressions_before", "Impressions_after"], var_name="Period", value_name="Impressions")
                         impressions_melted["Period"] = impressions_melted["Period"].str.replace("Impressions_", "")
                         fig_impressions = px.bar(impressions_melted, x="Topic", y="Impressions", color="Period", barmode="group", title="Total Impressions per Topic", labels={"Impressions": "Total Impressions"}, text_auto=True)
                         fig_impressions.update_traces(textposition='outside'); st.plotly_chart(fig_impressions, use_container_width=True)
                     except Exception as e: st.error(f"Error plotting Topic Impressions: {e}")
                else: st.warning("Cannot plot Topic Raw Impressions.")
            else: st.warning("No aggregated topic data for visualizations.")


            # --- Intent Visualization ---
            st.markdown("#### Performance by Search Intent")
            if not intent_agg_display.empty:
                st.markdown("##### Intent YoY % Change (Clicks & Impressions)")
                try:
                    intent_pct_melt = intent_agg_display.melt(id_vars='Intent', value_vars=['Clicks YoY %', 'Impressions YoY %'], var_name='Metric', value_name='YoY % Change')
                    intent_pct_melt['Metric'] = intent_pct_melt['Metric'].str.replace(' YoY %', '')
                    plot_data_intent_pct = intent_pct_melt.dropna(subset=['YoY % Change'])
                    if not plot_data_intent_pct.empty:
                        # Define a specific order for the Intent axis for better readability
                        intent_order = ["Informational", "Navigational", "Commercial", "Transactional", "Unknown"]
                        fig_intent_pct = px.bar(plot_data_intent_pct, x='Intent', y='YoY % Change', color='Metric',
                                                barmode='group', title='YoY % Change by Search Intent',
                                                labels={'YoY % Change': 'YoY Change (%)', 'Intent': 'Search Intent'},
                                                category_orders={"Intent": [i for i in intent_order if i in plot_data_intent_pct['Intent'].unique()]}, # Apply order
                                                text_auto='.1f')
                        fig_intent_pct.update_traces(textposition='outside')
                        st.plotly_chart(fig_intent_pct, use_container_width=True)
                    else: st.warning("No valid YoY % data available to plot for search intents.")
                except Exception as e: st.error(f"Error generating Intent YoY % Change plot: {e}"); st.error(traceback.format_exc())
            else: st.warning("No aggregated intent data available for visualization.")

            progress_bar.progress(100); status_text.text("Analysis Complete!")

        # --- Exception Handling ---
        except FileNotFoundError: st.error("Uploaded file not found."); status_text.text("Error: File not found.")
        except pd.errors.EmptyDataError: st.error("One or both CSV files are empty."); status_text.text("Error: Empty CSV file.")
        except KeyError as ke: st.error(f"Column missing: {ke}."); status_text.text(f"Error: Missing {ke}."); st.error(traceback.format_exc())
        except Exception as e: st.error(f"An unexpected error occurred: {e}"); st.error(traceback.format_exc()); status_text.text("Error during analysis.")
        finally:
            if 'progress_bar' in locals(): progress_bar.progress(100)
    else:
        st.info("Please upload both GSC CSV files ('Before' and 'After' periods).")

# --- Main Execution ---
def main():
    st.set_page_config(page_title="GSC Data Analysis", layout="wide")
    google_search_console_analysis_page()
    st.markdown("---"); st.markdown("GSC Analysis Tool")

if __name__ == "__main__":
    main()
