import streamlit as st
import pandas as pd
import numpy as np
import collections
from collections import Counter
import nltk
# NLTK is needed for stopwords in generate_topic_label and CountVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import traceback # For detailed error logging if needed

# ------------------------------------
# Helper Functions (Restored from Original)
# ------------------------------------

# Download NLTK data if necessary - Check added for robustness
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt') # Punkt might be needed implicitly by vectorizers/tokenizers
except LookupError:
    nltk.download('punkt')

# Load stopwords - Error handling added
try:
    nltk_stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading NLTK stopwords: {e}. Using a basic fallback list.")
    nltk_stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


# generate_topic_label restored from the *very first* code block provided.
def generate_topic_label(queries_in_topic):
    words = []
    for query in queries_in_topic:
        if isinstance(query, str):
             tokens = query.lower().split()
             words.extend(tokens)
    if words:
        freq = collections.Counter(words)
        for sw in list(nltk_stop_words):
             if sw in freq:
                 del freq[sw]
        common = freq.most_common(2)
        label = ", ".join([word for word, count in common])
        return label.capitalize() if label else "Topic Cluster"
    else:
         return "Unnamed Topic"

# Original CTR parsing logic helper
def parse_ctr_original(ctr):
    try:
        if isinstance(ctr, str) and "%" in ctr:
            return float(ctr.replace("%", "").replace(',', ''))
        else:
            return pd.to_numeric(ctr, errors='coerce')
    except Exception:
        return np.nan


# ------------------------------------
# GSC Analyzer Tool Function (Restored + New Charts)
# ------------------------------------

def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        The goal is to identify key topics that are contributing to your SEO performance.
        This tool lets you compare GSC query data from two different time periods. I recommend limiting to the top 1,000 queries as this can take awhile to process.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Classify queries into topics with descriptive labels using LDA.
        - Display the original merged data table with topic labels.
        - Aggregate metrics by topic, with an option to display more rows.
        - Visualize the YOY % change by topic for each metric.
        """
    )

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Reading CSV files...")
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            progress_bar.progress(10)

            status_text.text("Validating columns...")
            if "Top queries" not in df_before.columns or "Position" not in df_before.columns:
                st.error("The 'Before' CSV must contain 'Top queries' and 'Position' columns.")
                return
            if "Top queries" not in df_after.columns or "Position" not in df_after.columns:
                st.error("The 'After' CSV must contain 'Top queries' and 'Position' columns.")
                return
            progress_bar.progress(15)

            # --- Dashboard Summary ---
            status_text.text("Calculating dashboard summary...")
            st.markdown("## Dashboard Summary")
            df_before_renamed = df_before.rename(columns={"Top queries": "Query", "Position": "Average Position"})
            df_after_renamed = df_after.rename(columns={"Top queries": "Query", "Position": "Average Position"})
            progress_bar.progress(20)

            cols = st.columns(4)
            if "Clicks" in df_before_renamed.columns and "Clicks" in df_after_renamed.columns:
                total_clicks_before = pd.to_numeric(df_before_renamed["Clicks"], errors='coerce').fillna(0).sum()
                total_clicks_after = pd.to_numeric(df_after_renamed["Clicks"], errors='coerce').fillna(0).sum()
                overall_clicks_change = total_clicks_after - total_clicks_before
                overall_clicks_change_pct = (overall_clicks_change / total_clicks_before * 100) if total_clicks_before != 0 else 0
                cols[0].metric(label="Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
            else: cols[0].metric(label="Clicks Change", value="N/A")

            if "Impressions" in df_before_renamed.columns and "Impressions" in df_after_renamed.columns:
                total_impressions_before = pd.to_numeric(df_before_renamed["Impressions"], errors='coerce').fillna(0).sum()
                total_impressions_after = pd.to_numeric(df_after_renamed["Impressions"], errors='coerce').fillna(0).sum()
                overall_impressions_change = total_impressions_after - total_impressions_before
                overall_impressions_change_pct = (overall_impressions_change / total_impressions_before * 100) if total_impressions_before != 0 else 0
                cols[1].metric(label="Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
            else: cols[1].metric(label="Impressions Change", value="N/A")

            overall_avg_position_before = pd.to_numeric(df_before_renamed["Average Position"], errors='coerce').mean()
            overall_avg_position_after = pd.to_numeric(df_after_renamed["Average Position"], errors='coerce').mean()
            overall_position_change = overall_avg_position_before - overall_avg_position_after # Positive change means improved position (lower number)
            overall_position_change_pct = (-overall_position_change / overall_avg_position_before * 100) if pd.notna(overall_avg_position_before) and overall_avg_position_before != 0 else 0 # Use negative change for percentage improvement
            cols[2].metric(label="Avg. Position Change", value=f"{overall_position_change:.1f}", delta=f"{overall_position_change_pct:.1f}%")

            if "CTR" in df_before_renamed.columns and "CTR" in df_after_renamed.columns:
                df_before_renamed["CTR_parsed"] = df_before_renamed["CTR"].apply(parse_ctr_original)
                df_after_renamed["CTR_parsed"] = df_after_renamed["CTR"].apply(parse_ctr_original)
                overall_ctr_before = df_before_renamed["CTR_parsed"].mean()
                overall_ctr_after = df_after_renamed["CTR_parsed"].mean()
                overall_ctr_change = overall_ctr_after - overall_ctr_before
                overall_ctr_change_pct = (overall_ctr_change / overall_ctr_before * 100) if pd.notna(overall_ctr_before) and overall_ctr_before != 0 else 0
                cols[3].metric(label="CTR Change", value=f"{overall_ctr_change:.2f}", delta=f"{overall_ctr_change_pct:.1f}%")
            else: cols[3].metric(label="CTR Change", value="N/A")
            progress_bar.progress(30)

            status_text.text("Merging datasets (inner join)...")
            merged_df = pd.merge(df_before, df_after, on="Top queries", suffixes=("_before", "_after"), how="inner")
            progress_bar.progress(35)

            if merged_df.empty:
                st.warning("No common queries found between the two periods. Analysis cannot proceed.")
                status_text.text("Analysis stopped: No common queries.")
                progress_bar.progress(100)
                return

            status_text.text("Calculating YoY changes...")
            merged_df.rename(columns={"Top queries": "Query",
                                      "Position_before": "Average Position_before",
                                      "Position_after": "Average Position_after"}, inplace=True)

            merged_df["Average Position_before"] = pd.to_numeric(merged_df["Average Position_before"], errors='coerce')
            merged_df["Average Position_after"] = pd.to_numeric(merged_df["Average Position_after"], errors='coerce')
            merged_df["Position_YOY"] = merged_df["Average Position_before"] - merged_df["Average Position_after"] # Positive = improvement

            if "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns:
                merged_df["Clicks_before_num"] = pd.to_numeric(merged_df["Clicks_before"], errors='coerce').fillna(0)
                merged_df["Clicks_after_num"] = pd.to_numeric(merged_df["Clicks_after"], errors='coerce').fillna(0)
                merged_df["Clicks_YOY"] = merged_df["Clicks_after_num"] - merged_df["Clicks_before_num"]
            else: merged_df["Clicks_YOY"] = np.nan

            if "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns:
                merged_df["Impressions_before_num"] = pd.to_numeric(merged_df["Impressions_before"], errors='coerce').fillna(0)
                merged_df["Impressions_after_num"] = pd.to_numeric(merged_df["Impressions_after"], errors='coerce').fillna(0)
                merged_df["Impressions_YOY"] = merged_df["Impressions_after_num"] - merged_df["Impressions_before_num"]
            else: merged_df["Impressions_YOY"] = np.nan

            if "CTR_before" in merged_df.columns and "CTR_after" in merged_df.columns:
                merged_df["CTR_parsed_before"] = merged_df["CTR_before"].apply(parse_ctr_original)
                merged_df["CTR_parsed_after"] = merged_df["CTR_after"].apply(parse_ctr_original)
                merged_df["CTR_YOY"] = merged_df["CTR_parsed_after"] - merged_df["CTR_parsed_before"]
            else: merged_df["CTR_YOY"] = np.nan

            # Calculate Percentage Changes
            merged_df["Position_YOY_pct"] = merged_df.apply(lambda row: (-row["Position_YOY"] / row["Average Position_before"] * 100) if pd.notna(row["Position_YOY"]) and pd.notna(row["Average Position_before"]) and row["Average Position_before"] != 0 else np.nan, axis=1) # Negative YOY / Positive Before = Positive Percentage (Improvement)
            if "Clicks_YOY" in merged_df.columns and "Clicks_before_num" in merged_df.columns:
                merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: (row["Clicks_YOY"] / row["Clicks_before_num"] * 100) if pd.notna(row["Clicks_YOY"]) and pd.notna(row["Clicks_before_num"]) and row["Clicks_before_num"] != 0 else np.nan, axis=1)
            else: merged_df["Clicks_YOY_pct"] = np.nan
            if "Impressions_YOY" in merged_df.columns and "Impressions_before_num" in merged_df.columns:
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: (row["Impressions_YOY"] / row["Impressions_before_num"] * 100) if pd.notna(row["Impressions_YOY"]) and pd.notna(row["Impressions_before_num"]) and row["Impressions_before_num"] != 0 else np.nan, axis=1)
            else: merged_df["Impressions_YOY_pct"] = np.nan
            if "CTR_YOY" in merged_df.columns and "CTR_parsed_before" in merged_df.columns:
                 merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: (row["CTR_YOY"] / row["CTR_parsed_before"] * 100) if pd.notna(row["CTR_YOY"]) and pd.notna(row["CTR_parsed_before"]) and row["CTR_parsed_before"] != 0 else np.nan, axis=1)
            else: merged_df["CTR_YOY_pct"] = np.nan

            base_cols = ["Query", "Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"]
            if "Clicks_before" in merged_df.columns: base_cols += ["Clicks_before_num", "Clicks_after_num", "Clicks_YOY", "Clicks_YOY_pct"]
            if "Impressions_before" in merged_df.columns: base_cols += ["Impressions_before_num", "Impressions_after_num", "Impressions_YOY", "Impressions_YOY_pct"]
            if "CTR_before" in merged_df.columns:
                 if "CTR_parsed_before" in merged_df: base_cols += ["CTR_parsed_before", "CTR_parsed_after", "CTR_YOY", "CTR_YOY_pct"]
                 else: base_cols += ["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"] # Fallback if parsing failed somehow
            final_base_cols = [col for col in base_cols if col in merged_df.columns]
            merged_df_display_ready = merged_df[final_base_cols].copy()
            progress_bar.progress(40)

            # Rename columns for display (remove _num/_parsed)
            merged_df_display_ready.rename(columns={
                "Clicks_before_num": "Clicks_before",
                "Clicks_after_num": "Clicks_after",
                "Impressions_before_num": "Impressions_before",
                "Impressions_after_num": "Impressions_after",
                "CTR_parsed_before": "CTR_before",
                "CTR_parsed_after": "CTR_after",
            }, inplace=True)

            format_dict_merged = {}
            float_format = "{:.1f}"
            int_format = "{:,.0f}"
            pct_format = "{:+.1f}%" # Added + sign for changes
            pp_format = "{:+.2f}pp" # Added + sign for pp changes
            if "Average Position_before" in merged_df_display_ready.columns: format_dict_merged["Average Position_before"] = float_format
            if "Average Position_after" in merged_df_display_ready.columns: format_dict_merged["Average Position_after"] = float_format
            if "Position_YOY" in merged_df_display_ready.columns: format_dict_merged["Position_YOY"] = "{:+.1f}".format # Add sign to raw position change
            if "Clicks_before" in merged_df_display_ready.columns: format_dict_merged["Clicks_before"] = int_format
            if "Clicks_after" in merged_df_display_ready.columns: format_dict_merged["Clicks_after"] = int_format
            if "Clicks_YOY" in merged_df_display_ready.columns: format_dict_merged["Clicks_YOY"] = "{:+,.0f}".format # Add sign to raw click change
            if "Impressions_before" in merged_df_display_ready.columns: format_dict_merged["Impressions_before"] = int_format
            if "Impressions_after" in merged_df_display_ready.columns: format_dict_merged["Impressions_after"] = int_format
            if "Impressions_YOY" in merged_df_display_ready.columns: format_dict_merged["Impressions_YOY"] = "{:+,.0f}".format # Add sign to raw impression change
            if "CTR_before" in merged_df_display_ready.columns: format_dict_merged["CTR_before"] = "{:.2f}%".format # Keep CTR base as %
            if "CTR_after" in merged_df_display_ready.columns: format_dict_merged["CTR_after"] = "{:.2f}%".format # Keep CTR base as %
            if "CTR_YOY" in merged_df_display_ready.columns: format_dict_merged["CTR_YOY"] = pp_format # Use percentage points (pp)
            if "Position_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Position_YOY_pct"] = pct_format
            if "Clicks_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Clicks_YOY_pct"] = pct_format
            if "Impressions_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["Impressions_YOY_pct"] = pct_format
            if "CTR_YOY_pct" in merged_df_display_ready.columns: format_dict_merged["CTR_YOY_pct"] = pct_format

            status_text.text("Performing topic modeling (LDA)...")
            st.markdown("### Topic Classification and Combined Data")
            st.markdown("Data for queries present in both periods, classified by topic.")
            n_topics_gsc_lda = st.slider("Select number of topics for Query LDA:", min_value=2, max_value=15, value=5, key="lda_topics_gsc")

            queries = merged_df["Query"].astype(str).tolist()

            try:
                vectorizer_queries_lda = CountVectorizer(stop_words="english", min_df=2) # min_df added
                query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                if query_matrix_lda.shape[1] == 0:
                    st.warning("LDA Warning: No features found after applying stop words and min_df=2. Assigning 'Unclassified'. Try uploading more data or reducing min_df if appropriate.")
                    merged_df["Query_Topic"] = "Unclassified"
                    merged_df["Query_Topic_Label"] = -1
                else:
                    # Ensure n_topics isn't more than documents
                    actual_n_topics = min(n_topics_gsc_lda, query_matrix_lda.shape[0])
                    if actual_n_topics < n_topics_gsc_lda:
                         st.warning(f"Reduced number of topics to {actual_n_topics} (number of documents).")

                    # Ensure n_topics isn't less than 2 for LDA
                    if actual_n_topics < 2:
                        st.warning("Not enough documents/topics for meaningful LDA (need at least 2). Assigning 'Unclassified'.")
                        merged_df["Query_Topic"] = "Unclassified"
                        merged_df["Query_Topic_Label"] = -1
                    else:
                        lda_queries_model = LatentDirichletAllocation(n_components=actual_n_topics, random_state=42)
                        lda_queries_model.fit(query_matrix_lda)
                        query_topic_labels = lda_queries_model.transform(query_matrix_lda).argmax(axis=1)
                        merged_df["Query_Topic_Label"] = query_topic_labels
                        topic_labels_desc_queries = {}
                        for topic_idx in range(actual_n_topics):
                             topic_queries_lda = merged_df[merged_df["Query_Topic_Label"] == topic_idx]["Query"].tolist()
                             topic_labels_desc_queries[topic_idx] = generate_topic_label(topic_queries_lda)
                        merged_df["Query_Topic"] = merged_df["Query_Topic_Label"].apply(lambda x: topic_labels_desc_queries.get(x, f"Topic {x+1}"))

                        st.write("Identified Query Topics (Top Keywords):")
                        for topic_idx, topic_comp in enumerate(lda_queries_model.components_):
                            top_keyword_indices = topic_comp.argsort()[-10:][::-1]
                            topic_keywords = [feature_names_queries_lda[i] for i in top_keyword_indices if i < len(feature_names_queries_lda)] # Boundary check
                            desc_label = topic_labels_desc_queries.get(topic_idx, f"Topic {topic_idx+1}")
                            st.write(f"**{desc_label}:** {', '.join(topic_keywords)}")

            except Exception as lda_error:
                 st.error(f"Error during LDA Topic Modeling: {lda_error}")
                 st.error(traceback.format_exc())
                 st.warning("Assigning 'Unclassified' due to LDA error.")
                 merged_df["Query_Topic"] = "Unclassified"
                 merged_df["Query_Topic_Label"] = -1

            progress_bar.progress(50)

            if "Query_Topic" in merged_df.columns:
                 merged_df_display_ready.insert(1, "Query_Topic", merged_df["Query_Topic"])
            else:
                 merged_df_display_ready.insert(1, "Query_Topic", "Unclassified")
            st.markdown("#### Detailed Query Data by Topic")
            st.dataframe(merged_df_display_ready.style.format(format_dict_merged, na_rep="N/A"), use_container_width=True)


            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")
            if "Query_Topic" not in merged_df.columns or merged_df["Query_Topic"].nunique() == 0 or merged_df["Query_Topic"].isnull().all():
                 st.error("Cannot aggregate metrics: Topic modeling failed or produced no valid topics.")
                 return

            agg_dict = {}
            # Use the original numeric columns for aggregation
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
            # Rename columns AFTER aggregation
            aggregated.rename(columns={"Query_Topic": "Topic",
                                      "Clicks_before_num": "Clicks_before",
                                      "Clicks_after_num": "Clicks_after",
                                      "Impressions_before_num": "Impressions_before",
                                      "Impressions_after_num": "Impressions_after",
                                      "CTR_parsed_before": "CTR_before",
                                      "CTR_parsed_after": "CTR_after"}, inplace=True)

            # Recalculate percentages based on aggregated values
            aggregated["Position_YOY_pct"] = aggregated.apply(lambda row: (-row["Position_YOY"] / row["Average Position_before"] * 100) if pd.notna(row.get("Position_YOY")) and pd.notna(row.get("Average Position_before")) and row.get("Average Position_before") != 0 else np.nan, axis=1)
            if "Clicks_before" in aggregated.columns and "Clicks_YOY" in aggregated.columns:
                aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda row: (row["Clicks_YOY"] / row["Clicks_before"] * 100) if pd.notna(row.get("Clicks_YOY")) and pd.notna(row.get("Clicks_before")) and row.get("Clicks_before") != 0 else np.nan, axis=1)
            else: aggregated["Clicks_YOY_pct"] = np.nan
            if "Impressions_before" in aggregated.columns and "Impressions_YOY" in aggregated.columns:
                aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda row: (row["Impressions_YOY"] / row["Impressions_before"] * 100) if pd.notna(row.get("Impressions_YOY")) and pd.notna(row.get("Impressions_before")) and row.get("Impressions_before") != 0 else np.nan, axis=1)
            else: aggregated["Impressions_YOY_pct"] = np.nan
            if "CTR_before" in aggregated.columns and "CTR_YOY" in aggregated.columns:
                aggregated["CTR_YOY_pct"] = aggregated.apply(lambda row: (row["CTR_YOY"] / row["CTR_before"] * 100) if pd.notna(row.get("CTR_YOY")) and pd.notna(row.get("CTR_before")) and row.get("CTR_before") != 0 else np.nan, axis=1)
            else: aggregated["CTR_YOY_pct"] = np.nan
            progress_bar.progress(75)

            new_order = ["Topic"]
            if "Average Position_before" in aggregated.columns: new_order.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
            if "Clicks_before" in aggregated.columns: new_order.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
            if "Impressions_before" in aggregated.columns: new_order.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
            if "CTR_before" in aggregated.columns: new_order.extend(["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"])
            final_agg_order = [col for col in new_order if col in aggregated.columns]
            aggregated_display_final = aggregated[final_agg_order].copy()

            format_dict_agg = {}
            # Use the same formatting as the detailed table
            if "Average Position_before" in aggregated_display_final.columns: format_dict_agg["Average Position_before"] = float_format
            if "Average Position_after" in aggregated_display_final.columns: format_dict_agg["Average Position_after"] = float_format
            if "Position_YOY" in aggregated_display_final.columns: format_dict_agg["Position_YOY"] = "{:+.1f}".format
            if "Clicks_before" in aggregated_display_final.columns: format_dict_agg["Clicks_before"] = int_format
            if "Clicks_after" in aggregated_display_final.columns: format_dict_agg["Clicks_after"] = int_format
            if "Clicks_YOY" in aggregated_display_final.columns: format_dict_agg["Clicks_YOY"] = "{:+,.0f}".format
            if "Impressions_before" in aggregated_display_final.columns: format_dict_agg["Impressions_before"] = int_format
            if "Impressions_after" in aggregated_display_final.columns: format_dict_agg["Impressions_after"] = int_format
            if "Impressions_YOY" in aggregated_display_final.columns: format_dict_agg["Impressions_YOY"] = "{:+,.0f}".format
            if "CTR_before" in aggregated_display_final.columns: format_dict_agg["CTR_before"] = "{:.2f}%".format
            if "CTR_after" in aggregated_display_final.columns: format_dict_agg["CTR_after"] = "{:.2f}%".format
            if "CTR_YOY" in aggregated_display_final.columns: format_dict_agg["CTR_YOY"] = pp_format
            if "Position_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Position_YOY_pct"] = pct_format
            if "Clicks_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Clicks_YOY_pct"] = pct_format
            if "Impressions_YOY_pct" in aggregated_display_final.columns: format_dict_agg["Impressions_YOY_pct"] = pct_format
            if "CTR_YOY_pct" in aggregated_display_final.columns: format_dict_agg["CTR_YOY_pct"] = pct_format

            display_count = st.number_input(
                "Number of aggregated topics to display:", min_value=1,
                value=min(10, aggregated_display_final.shape[0]) if not aggregated_display_final.empty else 1,
                max_value=aggregated_display_final.shape[0] if not aggregated_display_final.empty else 1, step=1 )
            st.dataframe(aggregated_display_final.head(display_count).style.format(format_dict_agg, na_rep="N/A"), use_container_width=True)
            progress_bar.progress(80)

            # --- Step 6: Visualization ---
            status_text.text("Generating visualizations...")

            # Allow topic selection (keep this refinement)
            st.markdown("---") # Separator before charts
            st.markdown("### Visualizations by Topic")
            available_topics = aggregated_display_final["Topic"].unique().tolist()
            available_topics = [t for t in available_topics if pd.notna(t)] # Filter out potential NaN topics
            if not available_topics:
                st.warning("No valid topics found for visualization.")
                return # Stop if no topics to visualize

            selected_topics = st.multiselect(
                "Select topics to display on charts:",
                options=available_topics,
                default=available_topics
            )

            # Filter the aggregated data for selected topics
            aggregated_filtered = aggregated_display_final[aggregated_display_final['Topic'].isin(selected_topics)]

            if aggregated_filtered.empty:
                st.warning("No data available for the selected topics.")
                return # Stop if selection results in empty data

            # --- CHART 1: YoY % Change (Original) ---
            st.markdown("#### YoY % Change")
            vis_data_pct = []
            for idx, row in aggregated_filtered.iterrows():
                topic = row["Topic"]
                # Position: Note the negative sign because lower position is better
                if "Position_YOY_pct" in row and pd.notna(row["Position_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Avg. Position", "YOY % Change": row["Position_YOY_pct"]})
                if "Clicks_YOY_pct" in row and pd.notna(row["Clicks_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Clicks", "YOY % Change": row["Clicks_YOY_pct"]})
                if "Impressions_YOY_pct" in row and pd.notna(row["Impressions_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "Impressions", "YOY % Change": row["Impressions_YOY_pct"]})
                # CTR: Use the raw percentage point change for better comparison scale? Or stick to % change? Let's use % change for consistency with others.
                if "CTR_YOY_pct" in row and pd.notna(row["CTR_YOY_pct"]): vis_data_pct.append({"Topic": topic, "Metric": "CTR", "YOY % Change": row["CTR_YOY_pct"]})


            if not vis_data_pct:
                 st.warning("No YoY % change data available to plot for the selected topics.")
            else:
                vis_df_pct = pd.DataFrame(vis_data_pct)
                try:
                    fig_pct = px.bar(vis_df_pct, x="Topic", y="YOY % Change", color="Metric", barmode="group",
                                 title="YOY % Change by Topic for Each Metric",
                                 labels={"YOY % Change": "YoY Change (%)", "Topic": "Topic"},
                                 text_auto='.1f') # Show values on bars
                    fig_pct.update_traces(textposition='outside')
                    st.plotly_chart(fig_pct, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating YoY % Change plot: {e}")
                    st.error(traceback.format_exc())

            # --- CHART 2: Raw Clicks (NEW) ---
            st.markdown("#### Raw Clicks (Before vs After)")
            if "Clicks_before" in aggregated_filtered.columns and "Clicks_after" in aggregated_filtered.columns:
                try:
                    # Select and melt data for clicks plot
                    clicks_df = aggregated_filtered[["Topic", "Clicks_before", "Clicks_after"]]
                    clicks_melted = clicks_df.melt(id_vars="Topic",
                                                   value_vars=["Clicks_before", "Clicks_after"],
                                                   var_name="Period", value_name="Clicks")
                    # Clean up period names
                    clicks_melted["Period"] = clicks_melted["Period"].str.replace("Clicks_", "")

                    fig_clicks = px.bar(clicks_melted, x="Topic", y="Clicks", color="Period",
                                        barmode="group", title="Total Clicks per Topic (Before vs After)",
                                        labels={"Clicks": "Total Clicks", "Topic": "Topic"},
                                        text_auto=True) # Show values on bars, auto-format (usually integer)
                    fig_clicks.update_traces(textposition='outside')
                    st.plotly_chart(fig_clicks, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating Raw Clicks plot: {e}")
                    st.error(traceback.format_exc())
            else:
                st.warning("Cannot plot Raw Clicks: 'Clicks_before' or 'Clicks_after' column missing in aggregated data.")


            # --- CHART 3: Raw Impressions (NEW) ---
            st.markdown("#### Raw Impressions (Before vs After)")
            if "Impressions_before" in aggregated_filtered.columns and "Impressions_after" in aggregated_filtered.columns:
                try:
                    # Select and melt data for impressions plot
                    impressions_df = aggregated_filtered[["Topic", "Impressions_before", "Impressions_after"]]
                    impressions_melted = impressions_df.melt(id_vars="Topic",
                                                            value_vars=["Impressions_before", "Impressions_after"],
                                                            var_name="Period", value_name="Impressions")
                    # Clean up period names
                    impressions_melted["Period"] = impressions_melted["Period"].str.replace("Impressions_", "")

                    fig_impressions = px.bar(impressions_melted, x="Topic", y="Impressions", color="Period",
                                             barmode="group", title="Total Impressions per Topic (Before vs After)",
                                             labels={"Impressions": "Total Impressions", "Topic": "Topic"},
                                             text_auto=True) # Show values on bars
                    fig_impressions.update_traces(textposition='outside')
                    st.plotly_chart(fig_impressions, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating Raw Impressions plot: {e}")
                    st.error(traceback.format_exc())
            else:
                st.warning("Cannot plot Raw Impressions: 'Impressions_before' or 'Impressions_after' column missing in aggregated data.")


            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Uploaded file not found. Please re-upload.")
            status_text.text("Error: File not found.")
        except pd.errors.EmptyDataError:
            st.error("One or both uploaded CSV files are empty.")
            status_text.text("Error: Empty CSV file.")
        except KeyError as ke:
             st.error(f"Column missing or named incorrectly: {ke}. Required: 'Top queries', 'Position'. Others like 'Clicks', 'Impressions', 'CTR' are optional but needed for full analysis.")
             status_text.text(f"Error: Missing column {ke}.")
             st.error(traceback.format_exc())
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.error(traceback.format_exc())
            status_text.text("Error during analysis.")
        finally:
            if 'progress_bar' in locals():
                 progress_bar.progress(100)
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
    google_search_console_analysis_page()
    st.markdown("---") # Add a separator line
    # Add the link at the bottom
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
