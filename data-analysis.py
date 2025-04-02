import streamlit as st
import pandas as pd
import numpy as np
import collections
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

# ------------------------------------
# Helper Functions for GSC Tool
# ------------------------------------

# Download necessary NLTK data if not already present
# Using LookupError which is raised by nltk.data.find() when resource not found
try:
    nltk.data.find('corpora/stopwords')
    # st.write("Stopwords found.") # Optional: for debugging
except LookupError: # CORRECTED: Use LookupError
    # st.write("Stopwords not found, downloading...") # Optional: for debugging
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
    # st.write("Punkt found.") # Optional: for debugging
except LookupError: # CORRECTED: Use LookupError
    # st.write("Punkt not found, downloading...") # Optional: for debugging
    nltk.download('punkt')

# Load stopwords once (after ensuring they are downloaded)
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Failed to load NLTK stopwords after download attempt: {e}")
    # Provide a basic fallback or stop execution
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    st.warning("Using a basic default list of stopwords as NLTK list failed to load.")


def generate_topic_label(queries_in_topic):
    """Generates a simple topic label from the most common words in a list of queries."""
    words = []
    for query in queries_in_topic:
        # Ensure query is a string before processing
        if isinstance(query, str):
            tokens = query.lower().split()
            # Use the pre-loaded stop_words set
            filtered = [t for t in tokens if t not in stop_words and t.isalnum()] # Added isalnum() check
            words.extend(filtered)
        # Optionally handle non-string inputs, e.g., log a warning or skip
        # else:
        #     st.warning(f"Skipping non-string query in topic labeling: {query}")

    if words:
        freq = collections.Counter(words)
        common = freq.most_common(2) # Top 2 most common words
        label = ", ".join([word for word, count in common])
        return label.capitalize()
    else:
        # Handle cases where no valid words are found after filtering
        # Maybe return the first query or a generic label
        if queries_in_topic:
             # Fallback: return first query if available and is string
             first_query = queries_in_topic[0]
             return str(first_query)[:30] + "..." if isinstance(first_query, str) else "Topic Cluster"
        else:
            return "Unnamed Topic" # Or some other placeholder


# ------------------------------------
# GSC Analyzer Tool Function
# ------------------------------------

def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        Identify key topics contributing to your SEO performance by comparing GSC query data from two different time periods.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Classify queries into topics using Latent Dirichlet Allocation (LDA).
        - Display the original merged data table with topic labels.
        - Aggregate metrics by topic.
        - Visualize the Year-over-Year (YoY) % change by topic for each metric.
        *Note: Ensure your CSVs contain 'Top queries' and 'Position' columns. Other typical GSC columns like 'Clicks', 'Impressions', 'CTR' are optional but recommended.*
        """
    )

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        # Initialize the progress bar
        progress_bar = st.progress(0)
        status_text = st.empty() # Placeholder for status updates

        try:
            status_text.text("Reading CSV files...")
            # Step 1: Read the original CSV files
            # Use low_memory=False for potentially mixed type columns which GSC exports sometimes have
            df_before = pd.read_csv(uploaded_file_before, low_memory=False)
            df_after = pd.read_csv(uploaded_file_after, low_memory=False)
            progress_bar.progress(10)

            status_text.text("Validating columns...")
            # Step 2: Check required columns and rename
            required_before = {"Top queries", "Position"}
            required_after = {"Top queries", "Position"}
            if not required_before.issubset(df_before.columns):
                st.error(f"The 'Before' CSV must contain columns: {', '.join(required_before)}")
                return
            if not required_after.issubset(df_after.columns):
                st.error(f"The 'After' CSV must contain columns: {', '.join(required_after)}")
                return

            # Standardize column names early
            df_before.rename(columns={"Top queries": "Query", "Position": "Average Position"}, inplace=True)
            df_after.rename(columns={"Top queries": "Query", "Position": "Average Position"}, inplace=True)
            progress_bar.progress(15)

            # --- Dashboard Summary ---
            status_text.text("Calculating dashboard summary...")
            st.markdown("## Overall Performance Change (YoY)")

            # Helper function to safely calculate percentage change
            def calculate_pct_change(val_after, val_before):
                # Ensure inputs are numeric before division
                val_after = pd.to_numeric(val_after, errors='coerce')
                val_before = pd.to_numeric(val_before, errors='coerce')
                if pd.isna(val_before) or pd.isna(val_after) or val_before == 0:
                    return 0.0 # Or np.nan if you prefer to show blanks
                return ((val_after - val_before) / val_before) * 100

            # Helper function to parse CTR (handles strings with '%' and numbers)
            def parse_ctr(ctr_value):
                if pd.isna(ctr_value):
                    return np.nan
                if isinstance(ctr_value, str):
                    try:
                        # Handle potential commas in string numbers if GSC export includes them
                        return float(ctr_value.strip().replace('%', '').replace(',', '')) / 100.0
                    except ValueError:
                        return np.nan # Handle cases where string conversion fails
                elif isinstance(ctr_value, (int, float)):
                     # Assume GSC provides CTR as percentage (e.g., 5.5 for 5.5%)
                    return float(ctr_value) / 100.0
                return np.nan # Fallback for unexpected types

            cols = st.columns(4)
            metrics_calculated = 0

            # Clicks
            if "Clicks" in df_before.columns and "Clicks" in df_after.columns:
                # Handle potential non-numeric values like '<10' if present
                df_before["Clicks"] = pd.to_numeric(df_before["Clicks"], errors='coerce').fillna(0)
                df_after["Clicks"] = pd.to_numeric(df_after["Clicks"], errors='coerce').fillna(0)
                total_clicks_before = df_before["Clicks"].sum()
                total_clicks_after = df_after["Clicks"].sum()
                overall_clicks_change = total_clicks_after - total_clicks_before
                overall_clicks_change_pct = calculate_pct_change(total_clicks_after, total_clicks_before)
                cols[0].metric(label="Total Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
                metrics_calculated +=1
            else:
                cols[0].metric(label="Total Clicks Change", value="N/A", delta="Missing Data")

            # Impressions
            if "Impressions" in df_before.columns and "Impressions" in df_after.columns:
                df_before["Impressions"] = pd.to_numeric(df_before["Impressions"], errors='coerce').fillna(0)
                df_after["Impressions"] = pd.to_numeric(df_after["Impressions"], errors='coerce').fillna(0)
                total_impressions_before = df_before["Impressions"].sum()
                total_impressions_after = df_after["Impressions"].sum()
                overall_impressions_change = total_impressions_after - total_impressions_before
                overall_impressions_change_pct = calculate_pct_change(total_impressions_after, total_impressions_before)
                cols[1].metric(label="Total Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
                metrics_calculated +=1
            else:
                cols[1].metric(label="Total Impressions Change", value="N/A", delta="Missing Data")

            # Average Position
            df_before["Average Position"] = pd.to_numeric(df_before["Average Position"], errors='coerce')
            df_after["Average Position"] = pd.to_numeric(df_after["Average Position"], errors='coerce')

            # Calculate weighted average position if impressions available, handle NaNs
            valid_pos_before = df_before.dropna(subset=["Average Position"])
            if "Impressions" in valid_pos_before.columns and valid_pos_before["Impressions"].sum() > 0:
                 valid_pos_before_weighted = valid_pos_before.dropna(subset=["Impressions"])
                 if not valid_pos_before_weighted.empty:
                      overall_avg_position_before = np.average(valid_pos_before_weighted["Average Position"], weights=valid_pos_before_weighted["Impressions"])
                 else:
                      overall_avg_position_before = valid_pos_before["Average Position"].mean() # Fallback if no impressions
            else:
                 overall_avg_position_before = valid_pos_before["Average Position"].mean() if not valid_pos_before.empty else np.nan

            valid_pos_after = df_after.dropna(subset=["Average Position"])
            if "Impressions" in valid_pos_after.columns and valid_pos_after["Impressions"].sum() > 0:
                 valid_pos_after_weighted = valid_pos_after.dropna(subset=["Impressions"])
                 if not valid_pos_after_weighted.empty:
                      overall_avg_position_after = np.average(valid_pos_after_weighted["Average Position"], weights=valid_pos_after_weighted["Impressions"])
                 else:
                      overall_avg_position_after = valid_pos_after["Average Position"].mean() # Fallback if no impressions
            else:
                 overall_avg_position_after = valid_pos_after["Average Position"].mean() if not valid_pos_after.empty else np.nan


            if pd.notna(overall_avg_position_before) and pd.notna(overall_avg_position_after):
                 overall_position_change = overall_avg_position_after - overall_avg_position_before # Lower is better
                 cols[2].metric(label="Avg. Position Change", value=f"{overall_avg_position_after:.1f}", delta=f"{overall_position_change:.1f}", delta_color="inverse")
            else:
                 cols[2].metric(label="Avg. Position Change", value="N/A", delta="Cannot calc.")
            metrics_calculated +=1


            # CTR
            if "CTR" in df_before.columns and "CTR" in df_after.columns:
                df_before["CTR_parsed"] = df_before["CTR"].apply(parse_ctr)
                df_after["CTR_parsed"] = df_after["CTR"].apply(parse_ctr)

                # Calculate weighted average CTR if impressions available, handle NaNs
                valid_ctr_before = df_before.dropna(subset=["CTR_parsed"])
                if "Impressions" in valid_ctr_before.columns and valid_ctr_before["Impressions"].sum() > 0:
                     valid_ctr_before_weighted = valid_ctr_before.dropna(subset=["Impressions"])
                     if not valid_ctr_before_weighted.empty:
                          overall_ctr_before = np.average(valid_ctr_before_weighted["CTR_parsed"], weights=valid_ctr_before_weighted["Impressions"])
                     else:
                          overall_ctr_before = valid_ctr_before["CTR_parsed"].mean()
                else:
                     overall_ctr_before = valid_ctr_before["CTR_parsed"].mean() if not valid_ctr_before.empty else np.nan

                valid_ctr_after = df_after.dropna(subset=["CTR_parsed"])
                if "Impressions" in valid_ctr_after.columns and valid_ctr_after["Impressions"].sum() > 0:
                     valid_ctr_after_weighted = valid_ctr_after.dropna(subset=["Impressions"])
                     if not valid_ctr_after_weighted.empty:
                          overall_ctr_after = np.average(valid_ctr_after_weighted["CTR_parsed"], weights=valid_ctr_after_weighted["Impressions"])
                     else:
                          overall_ctr_after = valid_ctr_after["CTR_parsed"].mean()
                else:
                     overall_ctr_after = valid_ctr_after["CTR_parsed"].mean() if not valid_ctr_after.empty else np.nan

                if pd.notna(overall_ctr_before) and pd.notna(overall_ctr_after):
                     overall_ctr_change = overall_ctr_after - overall_ctr_before
                     # Display CTR as percentage, change in percentage points
                     cols[3].metric(label="Avg. CTR Change", value=f"{overall_ctr_after*100:.2f}%", delta=f"{overall_ctr_change*100:.2f}pp") # pp = percentage points
                else:
                     cols[3].metric(label="Avg. CTR Change", value="N/A", delta="Cannot calc.")
                metrics_calculated +=1
            else:
                cols[3].metric(label="Avg. CTR Change", value="N/A", delta="Missing Data")

            if metrics_calculated < 4:
                 st.warning("Some summary metrics could not be calculated due to missing columns (Clicks, Impressions, or CTR) or lack of numeric data.")
            progress_bar.progress(30)


            # Step 3: Merge Data for Detailed Analysis
            status_text.text("Merging datasets...")
            # Use outer merge to keep queries present in only one period
            merged_df = pd.merge(df_before, df_after, on="Query", suffixes=("_before", "_after"), how="outer")

            # Identify numeric columns for filling NaNs (excluding positions, CTR initially)
            numeric_cols_to_fill = []
            if "Clicks_before" in merged_df: numeric_cols_to_fill.append("Clicks_before")
            if "Clicks_after" in merged_df: numeric_cols_to_fill.append("Clicks_after")
            if "Impressions_before" in merged_df: numeric_cols_to_fill.append("Impressions_before")
            if "Impressions_after" in merged_df: numeric_cols_to_fill.append("Impressions_after")

            for col in numeric_cols_to_fill:
                 merged_df[col] = merged_df[col].fillna(0)
            # Keep NaNs for Position and CTR parsed columns to represent absence
            if "Average Position_before" in merged_df: merged_df["Average Position_before"] = pd.to_numeric(merged_df["Average Position_before"], errors='coerce')
            if "Average Position_after" in merged_df: merged_df["Average Position_after"] = pd.to_numeric(merged_df["Average Position_after"], errors='coerce')
            if "CTR_parsed_before" in merged_df: merged_df["CTR_parsed_before"] = merged_df["CTR_parsed_before"] # Already parsed
            if "CTR_parsed_after" in merged_df: merged_df["CTR_parsed_after"] = merged_df["CTR_parsed_after"] # Already parsed

            progress_bar.progress(35)

            # Calculate YOY changes
            status_text.text("Calculating YoY changes...")
            # Note: Subtracting with NaNs results in NaN, which is desired here
            merged_df["Position_YOY"] = merged_df["Average Position_after"] - merged_df["Average Position_before"] # Lower is better
            if "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns:
                merged_df["Clicks_YOY"] = merged_df["Clicks_after"] - merged_df["Clicks_before"]
            if "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns:
                merged_df["Impressions_YOY"] = merged_df["Impressions_after"] - merged_df["Impressions_before"]
            if "CTR_parsed_before" in merged_df.columns and "CTR_parsed_after" in merged_df.columns:
                merged_df["CTR_YOY"] = merged_df["CTR_parsed_after"] - merged_df["CTR_parsed_before"]

            # Calculate YOY percentage changes - use helper function which handles NaNs/zeros
            merged_df["Position_YOY_pct"] = merged_df.apply(
                lambda row: calculate_pct_change(row["Average Position_after"], row["Average Position_before"]), axis=1
            )
            if "Clicks_YOY" in merged_df.columns:
                 merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Clicks_after"], row["Clicks_before"]), axis=1)
            if "Impressions_YOY" in merged_df.columns:
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Impressions_after"], row["Impressions_before"]), axis=1)
            if "CTR_YOY" in merged_df.columns:
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["CTR_parsed_after"], row["CTR_parsed_before"]), axis=1)

            # Define columns to display in the merged table
            display_cols_merged = ["Query"]
            # Add columns if they exist in the merged_df
            if "Average Position_before" in merged_df: display_cols_merged.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
            if "Clicks_before" in merged_df: display_cols_merged.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
            if "Impressions_before" in merged_df: display_cols_merged.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
            # Display CTR as percentages using the parsed values
            if "CTR_parsed_before" in merged_df:
                merged_df["CTR %_before"] = merged_df["CTR_parsed_before"] * 100
                display_cols_merged.append("CTR %_before")
            if "CTR_parsed_after" in merged_df:
                 merged_df["CTR %_after"] = merged_df["CTR_parsed_after"] * 100
                 display_cols_merged.append("CTR %_after")
            if "CTR_YOY" in merged_df:
                 merged_df["CTR %_YOY"] = merged_df["CTR_YOY"] * 100 # Change in percentage points
                 display_cols_merged.append("CTR %_YOY")
            if "CTR_YOY_pct" in merged_df:
                 merged_df["CTR %_YOY_pct"] = merged_df["CTR_YOY_pct"] # This is the relative % change
                 display_cols_merged.append("CTR %_YOY_pct")


            # Filter display_cols_merged to only include columns actually present NOW
            display_cols_merged = [col for col in display_cols_merged if col in merged_df.columns]
            merged_df_display_base = merged_df[display_cols_merged].copy() # Select only the columns for display
            progress_bar.progress(40)


            # --- Define formatting for merged data table display ---
            format_dict_merged = {}
            pct_format = "{:,.1f}%" # Format for percentage changes
            pp_format = "{:,.1f}pp" # Format for percentage point changes
            float_format = "{:,.1f}"
            int_format = "{:,.0f}"

            if "Average Position_before" in merged_df_display_base.columns: format_dict_merged["Average Position_before"] = float_format
            if "Average Position_after" in merged_df_display_base.columns: format_dict_merged["Average Position_after"] = float_format
            if "Position_YOY" in merged_df_display_base.columns: format_dict_merged["Position_YOY"] = float_format
            if "Position_YOY_pct" in merged_df_display_base.columns: format_dict_merged["Position_YOY_pct"] = pct_format
            if "Clicks_before" in merged_df_display_base.columns: format_dict_merged["Clicks_before"] = int_format
            if "Clicks_after" in merged_df_display_base.columns: format_dict_merged["Clicks_after"] = int_format
            if "Clicks_YOY" in merged_df_display_base.columns: format_dict_merged["Clicks_YOY"] = int_format
            if "Clicks_YOY_pct" in merged_df_display_base.columns: format_dict_merged["Clicks_YOY_pct"] = pct_format
            if "Impressions_before" in merged_df_display_base.columns: format_dict_merged["Impressions_before"] = int_format
            if "Impressions_after" in merged_df_display_base.columns: format_dict_merged["Impressions_after"] = int_format
            if "Impressions_YOY" in merged_df_display_base.columns: format_dict_merged["Impressions_YOY"] = int_format
            if "Impressions_YOY_pct" in merged_df_display_base.columns: format_dict_merged["Impressions_YOY_pct"] = pct_format
            if "CTR %_before" in merged_df_display_base.columns: format_dict_merged["CTR %_before"] = pct_format
            if "CTR %_after" in merged_df_display_base.columns: format_dict_merged["CTR %_after"] = pct_format
            if "CTR %_YOY" in merged_df_display_base.columns: format_dict_merged["CTR %_YOY"] = pp_format # Display change in pp
            if "CTR %_YOY_pct" in merged_df_display_base.columns: format_dict_merged["CTR %_YOY_pct"] = pct_format # Display relative change


            # Step 4: Topic Classification using LDA
            status_text.text("Performing topic modeling (LDA)...")
            st.markdown("### Topic Modeling and Detailed Query Data")
            n_topics_gsc_lda = st.slider("Select number of topics for Query LDA:", min_value=2, max_value=20, value=8, key="lda_topics_gsc")

            # Ensure Query column exists and handle potential NaNs before LDA
            if "Query" not in merged_df.columns:
                 st.error("Critical Error: 'Query' column is missing after merge.")
                 return
            queries = merged_df["Query"].dropna().astype(str).tolist() # Drop NaN queries before LDA

            if not queries:
                 st.error("No valid queries found after dropping NaNs. Cannot perform topic modeling.")
                 return

            try:
                # Use stop_words loaded earlier
                vectorizer_queries_lda = CountVectorizer(stop_words=list(stop_words), max_df=0.9, min_df=3, token_pattern=r'\b[a-zA-Z]{2,}\b') # Require words with >= 2 letters
                query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                if query_matrix_lda.shape[0] == 0 or query_matrix_lda.shape[1] == 0:
                     st.warning("Could not create a valid document-term matrix for LDA. Check query data or adjust CountVectorizer parameters (min_df, max_df, stop_words). Assigning 'Unclassified'.")
                     # Add 'Query_Topic' column with default value to the original merged_df
                     merged_df["Query_Topic"] = "Unclassified"
                else:
                    # Ensure n_topics is not greater than number of documents
                    actual_n_topics = min(n_topics_gsc_lda, query_matrix_lda.shape[0])
                    if actual_n_topics < n_topics_gsc_lda:
                         st.warning(f"Reduced number of topics to {actual_n_topics} because it cannot exceed the number of documents ({query_matrix_lda.shape[0]}).")

                    if actual_n_topics < 2:
                         st.warning("Not enough documents/topics to perform LDA. Assigning 'Unclassified'.")
                         merged_df["Query_Topic"] = "Unclassified"
                    else:
                         lda_queries_model = LatentDirichletAllocation(n_components=actual_n_topics, random_state=42, max_iter=15, learning_method='online') # Added max_iter, changed method
                         lda_queries_model.fit(query_matrix_lda)

                         # Get topic distribution for the original list of non-NaN queries
                         query_topic_labels_array = lda_queries_model.transform(query_matrix_lda).argmax(axis=1)

                         # Create a mapping from query to topic label
                         query_to_topic_label = dict(zip(queries, query_topic_labels_array))

                         # Map topic labels back to the original merged_df, handling NaN queries
                         merged_df["Query_Topic_Label"] = merged_df["Query"].map(query_to_topic_label) # This will be NaN for original NaN queries

                         # Generate descriptive labels for topics
                         topic_labels_desc_queries = {}
                         for topic_idx in range(actual_n_topics):
                              # Get queries belonging to this topic *from the original list used for LDA*
                              topic_queries_lda = [q for q, label in query_to_topic_label.items() if label == topic_idx]
                              topic_labels_desc_queries[topic_idx] = generate_topic_label(topic_queries_lda)

                         # Apply descriptive labels, assign 'Unclassified' to original NaN queries or failed mappings
                         merged_df["Query_Topic"] = merged_df["Query_Topic_Label"].apply(
                             lambda x: topic_labels_desc_queries.get(x, f"Topic {int(x)+1}") if pd.notna(x) else "Unclassified"
                         )

                         # Display top keywords for each identified topic
                         st.write("Top keywords for identified query topics:")
                         for topic_idx, topic_comp in enumerate(lda_queries_model.components_):
                              top_keyword_indices = topic_comp.argsort()[-10:][::-1]
                              topic_keywords = [feature_names_queries_lda[i] for i in top_keyword_indices]
                              desc_label = topic_labels_desc_queries.get(topic_idx, f"Topic {topic_idx+1}")
                              st.write(f"**{desc_label}:** {', '.join(topic_keywords)}")

            except ValueError as ve:
                 st.error(f"Error during LDA vectorization or fitting: {ve}. This might happen if vocabulary is empty after filtering. Try adjusting CountVectorizer settings. Assigning 'Unclassified'.")
                 merged_df["Query_Topic"] = "Unclassified" # Assign placeholder
            except Exception as e:
                 st.error(f"An unexpected error occurred during topic modeling: {e}")
                 merged_df["Query_Topic"] = "Unclassified" # Assign placeholder

            progress_bar.progress(60)

            # --- Display Merged Data Table with Topic Labels ---
            st.markdown("#### Detailed Query Data with Topics")
            st.markdown("Use the filters to explore queries within specific topics.")

            # Add Topic filter - ensure 'Query_Topic' exists first
            if "Query_Topic" in merged_df.columns:
                topic_filter_options = ["All"] + sorted(merged_df["Query_Topic"].unique().astype(str).tolist())
                selected_topic_filter = st.selectbox("Filter by Topic:", options=topic_filter_options, key="topic_filter_merged")
            else:
                st.warning("Topic information is unavailable for filtering.")
                selected_topic_filter = "All" # Default if topics failed

            # Prepare the display dataframe again, inserting the topic column if it exists
            merged_df_display_filtered = merged_df_display_base.copy()
            if "Query_Topic" in merged_df.columns:
                 merged_df_display_filtered.insert(1, "Query_Topic", merged_df["Query_Topic"]) # Insert Topic column after Query
            else:
                 merged_df_display_filtered.insert(1, "Query_Topic", "N/A") # Add placeholder if failed

            # Apply filter
            if selected_topic_filter != "All" and "Query_Topic" in merged_df_display_filtered.columns:
                # Ensure comparison is robust (e.g., handle potential type mismatches if needed)
                merged_df_display_filtered = merged_df_display_filtered[merged_df_display_filtered["Query_Topic"].astype(str) == selected_topic_filter]


            # Use st.data_editor for interactive sorting/filtering in UI
            st.data_editor(
                merged_df_display_filtered,
                key="merged_data_editor",
                use_container_width=True,
                num_rows="dynamic", # Allow dynamic height
                column_config={ # Apply formatting within data_editor if possible, or use style after
                     col: st.column_config.NumberColumn(format=fmt.replace('%','%%')) # Need to escape % for NumberColumn format
                     for col, fmt in format_dict_merged.items() if col in merged_df_display_filtered.columns
                     # Add specific configs if needed, e.g., for links if URLs were included
                }
            )


            # Step 5: Aggregated Metrics by Topic
            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")

            # Check if Query_Topic column exists before grouping
            if "Query_Topic" not in merged_df.columns or merged_df["Query_Topic"].isnull().all():
                 st.error("Topic modeling failed or produced no valid topics, cannot aggregate by topic.")
                 return # Stop if no topics to aggregate by

            agg_dict = {}
            # Use np.nanmean to ignore NaNs during aggregation for positions/CTR
            if "Average Position_before" in merged_df: agg_dict["Average Position_before"] = lambda x: np.nanmean(x) if pd.notna(x).any() else np.nan
            if "Average Position_after" in merged_df: agg_dict["Average Position_after"] = lambda x: np.nanmean(x) if pd.notna(x).any() else np.nan
            # Sums for clicks/impressions
            if "Clicks_before" in merged_df: agg_dict["Clicks_before"] = "sum"
            if "Clicks_after" in merged_df: agg_dict["Clicks_after"] = "sum"
            if "Impressions_before" in merged_df: agg_dict["Impressions_before"] = "sum"
            if "Impressions_after" in merged_df: agg_dict["Impressions_after"] = "sum"
            # Use parsed CTR for aggregation (mean)
            if "CTR_parsed_before" in merged_df: agg_dict["CTR_parsed_before"] = lambda x: np.nanmean(x) if pd.notna(x).any() else np.nan
            if "CTR_parsed_after" in merged_df: agg_dict["CTR_parsed_after"] = lambda x: np.nanmean(x) if pd.notna(x).any() else np.nan


            # Group by topic, handle potential NaN topics if mapping failed for some rows
            aggregated = merged_df.groupby("Query_Topic", dropna=False).agg(agg_dict).reset_index() # dropna=False keeps NaN group if any
            aggregated.rename(columns={"Query_Topic": "Topic",
                                      "CTR_parsed_before": "CTR_before", # Rename back for consistency
                                      "CTR_parsed_after": "CTR_after"}, inplace=True)


            # Recalculate YOY and YOY % changes on aggregated data
            aggregated["Position_YOY"] = aggregated["Average Position_after"] - aggregated["Average Position_before"]
            aggregated["Position_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row["Average Position_after"], row["Average Position_before"]), axis=1)

            if "Clicks_before" in aggregated.columns:
                aggregated["Clicks_YOY"] = aggregated["Clicks_after"] - aggregated["Clicks_before"]
                aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row["Clicks_after"], row["Clicks_before"]), axis=1)
            if "Impressions_before" in aggregated.columns:
                aggregated["Impressions_YOY"] = aggregated["Impressions_after"] - aggregated["Impressions_before"]
                aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row["Impressions_after"], row["Impressions_before"]), axis=1)
            if "CTR_before" in aggregated.columns:
                aggregated["CTR_YOY"] = aggregated["CTR_after"] - aggregated["CTR_before"] # Change in percentage points
                aggregated["CTR_YOY_pct"] = aggregated.apply(lambda row: calculate_pct_change(row["CTR_after"], row["CTR_before"]), axis=1) # Relative % change

            progress_bar.progress(75)

            # Reorder columns for the aggregated table
            agg_display_order = ["Topic"]
            if "Average Position_before" in aggregated.columns: agg_display_order.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
            if "Clicks_before" in aggregated.columns: agg_display_order.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
            if "Impressions_before" in aggregated.columns: agg_display_order.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
            if "CTR_before" in aggregated.columns:
                # Display CTRs as percentages
                aggregated["CTR %_before"] = aggregated["CTR_before"] * 100
                aggregated["CTR %_after"] = aggregated["CTR_after"] * 100
                aggregated["CTR %_YOY"] = aggregated["CTR_YOY"] * 100 # Display change in pp
                aggregated["CTR %_YOY_pct"] = aggregated["CTR_YOY_pct"] # Relative change %
                agg_display_order.extend(["CTR %_before", "CTR %_after", "CTR %_YOY", "CTR %_YOY_pct"])

            # Filter agg_display_order to only include columns actually present
            agg_display_order = [col for col in agg_display_order if col in aggregated.columns]
            aggregated_display = aggregated[agg_display_order].copy() # Use .copy()

            # --- Define formatting for aggregated metrics display ---
            format_dict_agg = {}
            if "Average Position_before" in aggregated_display.columns: format_dict_agg["Average Position_before"] = float_format
            if "Average Position_after" in aggregated_display.columns: format_dict_agg["Average Position_after"] = float_format
            if "Position_YOY" in aggregated_display.columns: format_dict_agg["Position_YOY"] = float_format
            if "Position_YOY_pct" in aggregated_display.columns: format_dict_agg["Position_YOY_pct"] = pct_format
            if "Clicks_before" in aggregated_display.columns: format_dict_agg["Clicks_before"] = int_format
            if "Clicks_after" in aggregated_display.columns: format_dict_agg["Clicks_after"] = int_format
            if "Clicks_YOY" in aggregated_display.columns: format_dict_agg["Clicks_YOY"] = int_format
            if "Clicks_YOY_pct" in aggregated_display.columns: format_dict_agg["Clicks_YOY_pct"] = pct_format
            if "Impressions_before" in aggregated_display.columns: format_dict_agg["Impressions_before"] = int_format
            if "Impressions_after" in aggregated_display.columns: format_dict_agg["Impressions_after"] = int_format
            if "Impressions_YOY" in aggregated_display.columns: format_dict_agg["Impressions_YOY"] = int_format
            if "Impressions_YOY_pct" in aggregated_display.columns: format_dict_agg["Impressions_YOY_pct"] = pct_format
            if "CTR %_before" in aggregated_display.columns: format_dict_agg["CTR %_before"] = pct_format
            if "CTR %_after" in aggregated_display.columns: format_dict_agg["CTR %_after"] = pct_format
            if "CTR %_YOY" in aggregated_display.columns: format_dict_agg["CTR %_YOY"] = pp_format # Show pp change
            if "CTR %_YOY_pct" in aggregated_display.columns: format_dict_agg["CTR %_YOY_pct"] = pct_format # Show relative change

            # Use st.data_editor for the aggregated table too
            st.data_editor(
                aggregated_display,
                key="aggregated_data_editor",
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                     col: st.column_config.NumberColumn(format=fmt.replace('%','%%'))
                     for col, fmt in format_dict_agg.items() if col in aggregated_display.columns
                }
            )
            progress_bar.progress(80)


            # Step 6: Visualization - Grouped Bar Chart of YOY % Change by Topic for Each Metric
            status_text.text("Generating visualizations...")
            st.markdown("### YoY % Change by Topic")

            vis_data = []
             # Use the aggregated dataframe which has the calculated YOY % changes
            for idx, row in aggregated_display.iterrows(): # Iterate over the display DF
                topic = row["Topic"]
                # Use the YOY % columns calculated on aggregated data
                # Check existence and ensure value is not NaN before adding
                if "Position_YOY_pct" in row and pd.notna(row["Position_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Avg. Position % Change", "Value": row["Position_YOY_pct"]})
                if "Clicks_YOY_pct" in row and pd.notna(row["Clicks_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Clicks % Change", "Value": row["Clicks_YOY_pct"]})
                if "Impressions_YOY_pct" in row and pd.notna(row["Impressions_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Impressions % Change", "Value": row["Impressions_YOY_pct"]})
                # Use the correct column name 'CTR %_YOY_pct'
                if "CTR %_YOY_pct" in row and pd.notna(row["CTR %_YOY_pct"]):
                     vis_data.append({"Topic": topic, "Metric": "CTR % Change (Relative)", "Value": row["CTR %_YOY_pct"]})

            if not vis_data:
                st.warning("No data available for visualization. Check if YoY % changes could be calculated for any topics.")
            else:
                vis_df = pd.DataFrame(vis_data)

                # Allow user to select which metrics to plot
                available_metrics = sorted(vis_df["Metric"].unique().tolist())
                selected_metrics = st.multiselect("Select metrics to display on chart:", options=available_metrics, default=available_metrics)

                if selected_metrics:
                     vis_df_filtered = vis_df[vis_df["Metric"].isin(selected_metrics)]

                     # Determine plot type based on number of topics
                     num_unique_topics = len(aggregated_display["Topic"].unique())
                     chart_height = max(450, num_unique_topics * 35) # Dynamic height

                     try:
                         fig = px.bar(vis_df_filtered,
                                      y="Topic", # Horizontal bar chart often better for many categories
                                      x="Value",
                                      color="Metric",
                                      barmode="group",
                                      orientation='h', # Horizontal
                                      title="YoY % Change by Topic for Selected Metrics",
                                      labels={"Value": "YoY Change (%)", "Topic": "Topic"},
                                      height=chart_height,
                                      text_auto='.1f' # Display values on bars
                                     )
                         fig.update_layout(
                             yaxis={'categoryorder':'total ascending'}, # Sort topics by value summed across metrics
                             legend_title_text='Metric',
                             xaxis_title="YoY Change (%)",
                             yaxis_title="Topic"
                         )
                         fig.update_traces(textposition='outside')
                         st.plotly_chart(fig, use_container_width=True)
                     except Exception as plot_err:
                          st.error(f"Failed to generate plot: {plot_err}")

                else:
                     st.info("Select at least one metric to display the chart.")

            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Uploaded file not found. Please re-upload.")
        except pd.errors.EmptyDataError:
            st.error("One of the uploaded CSV files is empty or contains no data.")
        except KeyError as ke:
             st.error(f"A required column is missing or named incorrectly: {ke}. Please ensure 'Top queries' and 'Position' are present.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            # Optional: Add more detailed error logging here if needed
            import traceback
            st.error(traceback.format_exc()) # Show full traceback for debugging
        finally:
             # Ensure progress bar completes and status text is cleared or updated
             progress_bar.progress(100)
             # status_text.text("Analysis Finished.") # Or clear it: status_text.empty()

    else:
        st.info("Please upload both GSC CSV files ('Before' and 'After' periods) to start the analysis.")

# ------------------------------------
# Main Streamlit App Execution
# ------------------------------------
def main():
    st.set_page_config(
        page_title="GSC Data Analysis",
        layout="wide"
    )
    # Optional: Hide default Streamlit elements
    # hide_streamlit_elements = """<style>...</style>"""
    # st.markdown(hide_streamlit_elements, unsafe_allow_html=True)

    # Directly call the only tool function
    google_search_console_analysis_page()

    # Optional: Add a footer
    st.markdown("---")
    st.markdown("GSC Analysis Tool | Powered by Streamlit")

if __name__ == "__main__":
    main()
