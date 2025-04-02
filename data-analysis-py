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
# Best practice: Do this once, potentially outside the main execution flow
# if running in an environment where it persists, but including it here ensures it runs.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
# wordnet is needed if lemmatization were used, but generate_topic_label doesn't use it.
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')

# Load stopwords once
stop_words = set(stopwords.words('english'))

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
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
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
                if pd.isna(val_before) or pd.isna(val_after) or val_before == 0:
                    return 0.0
                return ((val_after - val_before) / val_before) * 100

            # Helper function to parse CTR (handles strings with '%' and numbers)
            def parse_ctr(ctr_value):
                if pd.isna(ctr_value):
                    return np.nan
                if isinstance(ctr_value, str):
                    try:
                        return float(ctr_value.strip().replace('%', '')) / 100.0
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
            # Weighted average position by impressions if available, otherwise simple mean
            if "Impressions" in df_before.columns and df_before["Impressions"].sum() > 0:
                 overall_avg_position_before = np.average(df_before["Average Position"].dropna(), weights=df_before["Impressions"].dropna())
            else:
                 overall_avg_position_before = df_before["Average Position"].mean()

            if "Impressions" in df_after.columns and df_after["Impressions"].sum() > 0:
                 overall_avg_position_after = np.average(df_after["Average Position"].dropna(), weights=df_after["Impressions"].dropna())
            else:
                 overall_avg_position_after = df_after["Average Position"].mean()

            overall_position_change = overall_avg_position_after - overall_avg_position_before # Lower is better
            # For position, delta_color="inverse" makes improvement (negative change) green
            cols[2].metric(label="Avg. Position Change", value=f"{overall_avg_position_after:.1f}", delta=f"{overall_position_change:.1f}", delta_color="inverse")
            metrics_calculated +=1


            # CTR
            if "CTR" in df_before.columns and "CTR" in df_after.columns:
                df_before["CTR_parsed"] = df_before["CTR"].apply(parse_ctr)
                df_after["CTR_parsed"] = df_after["CTR"].apply(parse_ctr)

                # Weighted average CTR by impressions if available and clicks non-zero
                if "Impressions" in df_before.columns and df_before["Impressions"].sum() > 0 and "Clicks" in df_before.columns:
                     valid_before = df_before.dropna(subset=["CTR_parsed", "Impressions"])
                     overall_ctr_before = np.average(valid_before["CTR_parsed"], weights=valid_before["Impressions"])
                else:
                     overall_ctr_before = df_before["CTR_parsed"].mean()

                if "Impressions" in df_after.columns and df_after["Impressions"].sum() > 0 and "Clicks" in df_after.columns:
                     valid_after = df_after.dropna(subset=["CTR_parsed", "Impressions"])
                     overall_ctr_after = np.average(valid_after["CTR_parsed"], weights=valid_after["Impressions"])
                else:
                     overall_ctr_after = df_after["CTR_parsed"].mean()

                overall_ctr_change = overall_ctr_after - overall_ctr_before
                overall_ctr_change_pct = calculate_pct_change(overall_ctr_after, overall_ctr_before)
                # Display CTR as percentage
                cols[3].metric(label="Avg. CTR Change", value=f"{overall_ctr_after*100:.2f}%", delta=f"{overall_ctr_change*100:.2f}pp") # pp = percentage points
                metrics_calculated +=1
            else:
                cols[3].metric(label="Avg. CTR Change", value="N/A", delta="Missing Data")

            if metrics_calculated < 4:
                 st.warning("Some summary metrics could not be calculated due to missing columns (Clicks, Impressions, or CTR).")
            progress_bar.progress(30)


            # Step 3: Merge Data for Detailed Analysis
            status_text.text("Merging datasets...")
            # Use outer merge to keep queries present in only one period
            merged_df = pd.merge(df_before, df_after, on="Query", suffixes=("_before", "_after"), how="outer")

            # Fill NaN values for metrics with 0 for calculations, but maybe handle position differently?
            # For position, NaN means the query didn't exist in that period. We might leave it NaN.
            numeric_cols_to_fill = []
            if "Clicks_before" in merged_df: numeric_cols_to_fill.append("Clicks_before")
            if "Clicks_after" in merged_df: numeric_cols_to_fill.append("Clicks_after")
            if "Impressions_before" in merged_df: numeric_cols_to_fill.append("Impressions_before")
            if "Impressions_after" in merged_df: numeric_cols_to_fill.append("Impressions_after")
            if "CTR_parsed_before" in merged_df: numeric_cols_to_fill.append("CTR_parsed_before")
            if "CTR_parsed_after" in merged_df: numeric_cols_to_fill.append("CTR_parsed_after")

            for col in numeric_cols_to_fill:
                 merged_df[col] = merged_df[col].fillna(0)
            # Position: Keep NaNs for now to indicate absence in a period
            if "Average Position_before" in merged_df: merged_df["Average Position_before"] = pd.to_numeric(merged_df["Average Position_before"], errors='coerce')
            if "Average Position_after" in merged_df: merged_df["Average Position_after"] = pd.to_numeric(merged_df["Average Position_after"], errors='coerce')

            progress_bar.progress(35)

            # Calculate YOY changes
            status_text.text("Calculating YoY changes...")
            merged_df["Position_YOY"] = merged_df["Average Position_after"] - merged_df["Average Position_before"] # Lower is better
            if "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns:
                merged_df["Clicks_YOY"] = merged_df["Clicks_after"] - merged_df["Clicks_before"]
            if "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns:
                merged_df["Impressions_YOY"] = merged_df["Impressions_after"] - merged_df["Impressions_before"]
            if "CTR_parsed_before" in merged_df.columns and "CTR_parsed_after" in merged_df.columns:
                merged_df["CTR_YOY"] = merged_df["CTR_parsed_after"] - merged_df["CTR_parsed_before"]

            # Calculate YOY percentage changes
            merged_df["Position_YOY_pct"] = merged_df.apply(
                lambda row: calculate_pct_change(row["Average Position_after"], row["Average Position_before"])
                if pd.notna(row["Average Position_before"]) and pd.notna(row["Average Position_after"]) else np.nan, axis=1
            )
            if "Clicks_YOY" in merged_df.columns:
                 merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Clicks_after"], row["Clicks_before"]), axis=1)
            if "Impressions_YOY" in merged_df.columns:
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["Impressions_after"], row["Impressions_before"]), axis=1)
            if "CTR_YOY" in merged_df.columns:
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: calculate_pct_change(row["CTR_parsed_after"], row["CTR_parsed_before"]), axis=1)

            # Define columns to display in the merged table
            display_cols_merged = ["Query"]
            if "Average Position_before" in merged_df: display_cols_merged.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
            if "Clicks_before" in merged_df: display_cols_merged.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
            if "Impressions_before" in merged_df: display_cols_merged.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
            # Display CTR as percentages
            if "CTR_parsed_before" in merged_df:
                merged_df["CTR %_before"] = merged_df["CTR_parsed_before"] * 100
                merged_df["CTR %_after"] = merged_df["CTR_parsed_after"] * 100
                merged_df["CTR %_YOY"] = merged_df["CTR_YOY"] * 100 # Change in percentage points
                merged_df["CTR %_YOY_pct"] = merged_df["CTR_YOY_pct"] # This is the relative % change
                display_cols_merged.extend(["CTR %_before", "CTR %_after", "CTR %_YOY", "CTR %_YOY_pct"])


            # Filter display_cols_merged to only include columns actually present
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

            queries = merged_df["Query"].astype(str).tolist() # Ensure queries are strings for vectorizer
            if not queries:
                 st.error("No queries found in the merged data to perform topic modeling.")
                 return

            try:
                vectorizer_queries_lda = CountVectorizer(stop_words="english", max_df=0.9, min_df=3) # Adjusted params
                query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                if query_matrix_lda.shape[0] == 0 or query_matrix_lda.shape[1] == 0:
                     st.warning("Could not create a valid document-term matrix for LDA. Check query data or adjust CountVectorizer parameters (min_df, max_df).")
                     merged_df["Query_Topic"] = "Topic Modeling Failed" # Assign placeholder
                else:
                    # Ensure n_topics is not greater than number of documents
                    actual_n_topics = min(n_topics_gsc_lda, query_matrix_lda.shape[0])
                    if actual_n_topics < n_topics_gsc_lda:
                         st.warning(f"Reduced number of topics to {actual_n_topics} because it cannot exceed the number of documents ({query_matrix_lda.shape[0]}).")

                    if actual_n_topics < 2:
                         st.warning("Not enough documents to perform LDA with at least 2 topics.")
                         merged_df["Query_Topic"] = "Topic Modeling Failed"
                    else:
                         lda_queries_model = LatentDirichletAllocation(n_components=actual_n_topics, random_state=42, max_iter=10) # Added max_iter
                         lda_queries_model.fit(query_matrix_lda)

                         query_topic_labels = lda_queries_model.transform(query_matrix_lda).argmax(axis=1)
                         merged_df["Query_Topic_Label"] = query_topic_labels

                         # Generate descriptive labels
                         topic_labels_desc_queries = {}
                         for topic_idx in range(actual_n_topics):
                              topic_queries_lda = merged_df[merged_df["Query_Topic_Label"] == topic_idx]["Query"].tolist()
                              topic_labels_desc_queries[topic_idx] = generate_topic_label(topic_queries_lda)
                         merged_df["Query_Topic"] = merged_df["Query_Topic_Label"].apply(lambda x: topic_labels_desc_queries.get(x, f"Topic {x+1}"))

                         # Display top keywords for each topic
                         st.write("Top keywords for identified query topics:")
                         for topic_idx, topic_comp in enumerate(lda_queries_model.components_):
                              top_keyword_indices = topic_comp.argsort()[-10:][::-1]
                              topic_keywords = [feature_names_queries_lda[i] for i in top_keyword_indices]
                              desc_label = topic_labels_desc_queries.get(topic_idx, f"Topic {topic_idx+1}")
                              st.write(f"**{desc_label}:** {', '.join(topic_keywords)}")

            except ValueError as ve:
                 st.error(f"Error during LDA vectorization or fitting: {ve}. This might happen if vocabulary is empty after filtering. Try adjusting min_df/max_df or checking input data.")
                 merged_df["Query_Topic"] = "Topic Modeling Error" # Assign placeholder
            except Exception as e:
                 st.error(f"An unexpected error occurred during topic modeling: {e}")
                 merged_df["Query_Topic"] = "Topic Modeling Error"

            progress_bar.progress(60)

            # --- Display Merged Data Table with Topic Labels ---
            st.markdown("#### Detailed Query Data with Topics")
            st.markdown("Use the filters to explore queries within specific topics.")

            # Add Topic filter
            topic_filter_options = ["All"] + sorted(merged_df["Query_Topic"].unique().tolist())
            selected_topic_filter = st.selectbox("Filter by Topic:", options=topic_filter_options, key="topic_filter_merged")

            merged_df_display_filtered = merged_df_display_base.copy()
            merged_df_display_filtered.insert(1, "Query_Topic", merged_df["Query_Topic"]) # Insert Topic column after Query

            if selected_topic_filter != "All":
                merged_df_display_filtered = merged_df_display_filtered[merged_df_display_filtered["Query_Topic"] == selected_topic_filter]


            st.dataframe(merged_df_display_filtered.style.format(format_dict_merged, na_rep="N/A")) # Display merged_df with formatting


            # Step 5: Aggregated Metrics by Topic
            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")
            agg_dict = {}
             # Use np.nanmean to ignore NaNs during aggregation for positions/CTR
            if "Average Position_before" in merged_df: agg_dict["Average Position_before"] = lambda x: np.nanmean(x) if not x.isnull().all() else np.nan
            if "Average Position_after" in merged_df: agg_dict["Average Position_after"] = lambda x: np.nanmean(x) if not x.isnull().all() else np.nan
            # Sums for clicks/impressions
            if "Clicks_before" in merged_df: agg_dict["Clicks_before"] = "sum"
            if "Clicks_after" in merged_df: agg_dict["Clicks_after"] = "sum"
            if "Impressions_before" in merged_df: agg_dict["Impressions_before"] = "sum"
            if "Impressions_after" in merged_df: agg_dict["Impressions_after"] = "sum"
            # Use parsed CTR for aggregation (mean)
            if "CTR_parsed_before" in merged_df: agg_dict["CTR_parsed_before"] = lambda x: np.nanmean(x) if not x.isnull().all() else np.nan
            if "CTR_parsed_after" in merged_df: agg_dict["CTR_parsed_after"] = lambda x: np.nanmean(x) if not x.isnull().all() else np.nan

            # Check if Query_Topic column exists before grouping
            if "Query_Topic" not in merged_df.columns:
                 st.error("Topic modeling failed, cannot aggregate by topic.")
                 return

            aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index()
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
            aggregated_display = aggregated[agg_display_order]

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


            # Allow sorting the aggregated table
            sort_col_options = aggregated_display.columns.tolist()
            default_sort_col = "Clicks_after" if "Clicks_after" in sort_col_options else "Topic"
            sort_column = st.selectbox("Sort aggregated table by:", options=sort_col_options, index=sort_col_options.index(default_sort_col), key="agg_sort_col")
            sort_ascending = st.checkbox("Sort Ascending", value=False, key="agg_sort_asc")

            aggregated_display_sorted = aggregated_display.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')


            st.dataframe(aggregated_display_sorted.style.format(format_dict_agg, na_rep="N/A"))
            progress_bar.progress(80)

            # Step 6: Visualization - Grouped Bar Chart of YOY % Change by Topic for Each Metric
            status_text.text("Generating visualizations...")
            st.markdown("### YoY % Change by Topic")

            vis_data = []
            for idx, row in aggregated.iterrows(): # Use original aggregated df before selecting display columns
                topic = row["Topic"]
                # Use the YOY % columns calculated on aggregated data
                if "Position_YOY_pct" in row and pd.notna(row["Position_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Avg. Position % Change", "Value": row["Position_YOY_pct"], "Color": "Position"})
                if "Clicks_YOY_pct" in row and pd.notna(row["Clicks_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Clicks % Change", "Value": row["Clicks_YOY_pct"], "Color": "Clicks"})
                if "Impressions_YOY_pct" in row and pd.notna(row["Impressions_YOY_pct"]):
                    vis_data.append({"Topic": topic, "Metric": "Impressions % Change", "Value": row["Impressions_YOY_pct"], "Color": "Impressions"})
                if "CTR %_YOY_pct" in aggregated_display.columns and pd.notna(row["CTR %_YOY_pct"]): # Check the display col name here
                     vis_data.append({"Topic": topic, "Metric": "CTR % Change (Relative)", "Value": row["CTR %_YOY_pct"], "Color": "CTR"})

            if not vis_data:
                st.warning("No data available for visualization. Check if YoY % changes could be calculated.")
            else:
                vis_df = pd.DataFrame(vis_data)

                # Allow user to select which metrics to plot
                available_metrics = vis_df["Metric"].unique().tolist()
                selected_metrics = st.multiselect("Select metrics to display on chart:", options=available_metrics, default=available_metrics)

                if selected_metrics:
                     vis_df_filtered = vis_df[vis_df["Metric"].isin(selected_metrics)]

                     # Determine plot type based on number of topics
                     num_unique_topics = len(aggregated["Topic"].unique())
                     chart_height = max(400, num_unique_topics * 30) # Dynamic height

                     fig = px.bar(vis_df_filtered,
                                  y="Topic", # Horizontal bar chart often better for many categories
                                  x="Value",
                                  color="Metric",
                                  barmode="group",
                                  orientation='h', # Horizontal
                                  title="YoY % Change by Topic for Selected Metrics",
                                  labels={"Value": "YoY Change (%)", "Topic": "Topic"},
                                  height=chart_height
                                 )
                     fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort topics by value
                     st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("Select at least one metric to display the chart.")

            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Uploaded file not found. Please re-upload.")
        except pd.errors.EmptyDataError:
            st.error("One of the uploaded CSV files is empty.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Optional: Add more detailed error logging here if needed
            # import traceback
            # st.error(traceback.format_exc())
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
    # You can optionally hide the default Streamlit elements if desired
    # hide_streamlit_elements = """
    #     <style>
    #     #MainMenu {visibility: hidden !important;}
    #     header {visibility: hidden !important;}
    #     footer {visibility: hidden !important;}
    #     [data-testid="stDecoration"] { display: none !important; }
    #     div.block-container {padding-top: 1rem;}
    #     </style>
    #     """
    # st.markdown(hide_streamlit_elements, unsafe_allow_html=True)

    # Directly call the only tool function
    google_search_console_analysis_page()

    # Optional: Add a footer if needed
    st.markdown("---")
    st.markdown("GSC Analysis Tool")

if __name__ == "__main__":
    main()
