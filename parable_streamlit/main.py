import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from app import sentiment_analysis, topic_modelling, entity_recognition, actionable_insights
import plotly.graph_objects as go
import plotly.express as px
#from semantic_search import source_docs, print_answer, search_index
#from streamlit_chat import message


def main():
    st.set_page_config(page_title='Parable - A Text to Insight Tool')

    st.title('Parable - A Text to Insight tool')

    st.sidebar.title('Choose an Analysis')

    analysis_choice = st.sidebar.selectbox(
        'Select an analysis:',
        ('Sentiment Analysis', 'Topic Modelling', 'Entity Recognition', 'Actionable Insights')
    )

    st.sidebar.markdown('---')

    st.sidebar.title('About')

    st.sidebar.info(
        'Parable is a text to insight tool leveraging LLMs to extract meaningul insights from textual data.' 
    )

    st.sidebar.title('Instructions')

    st.sidebar.success(
        'Upload a file in .csv, .txt, .doc, .docx, or .pdf format and click the "Analyse" button.'
    )

    st.sidebar.markdown('---')

    st.sidebar.title('Contact')

    st.sidebar.info(
        'For any questions or issues, please email us at support@textanalysis.com'
    )

    if analysis_choice == 'Sentiment Analysis':
        st.header('Sentiment Analysis')

        form = st.form(key='sentiment_analysis_form')
        uploaded_file = form.file_uploader('Upload a file in .csv, .txt, .doc, .docx, or .pdf format')

        custom_parameters = form.text_input('Add Custom Parameters')
        insight = form.text_input('Tell us a little bit about your data')

        submit_button = form.form_submit_button(label='Analyse')

        if submit_button and uploaded_file is not None:
            data = sentiment_analysis(uploaded_file, custom_parameters, insight)

            st.subheader("Summary")
            st.write(data["summary"])

            st.subheader("Positive ({})".format(len(data["positive_words"])))
            st.table([(word,) for word in data["positive_words"]])
            st.subheader("Negative ({})".format(len(data["negative_words"])))
            st.table([(word,) for word in data["negative_words"]])
            st.subheader("Neutral ({})".format(len(data["neutral_words"])))
            st.table([(word,) for word in data["neutral_words"]])

            new_data = {
                'Positive': len(data['positive_words']),
                'Negative': len(data['negative_words']),
                'Neutral': len(data['neutral_words'])
            }

            # Create the pie chart
            fig = go.Figure(data=[go.Pie(labels=list(new_data.keys()), values=list(new_data.values()))])

            # Set the title of the chart
            fig.update_layout(title='Analysis - Pie Chart')

            # Render the chart using Streamlit
            st.plotly_chart(fig)

            # Display summary and custom parameters

            st.subheader("Custom Q/A")
            st.write(data["custom_parameters"])
                        


    elif analysis_choice == 'Topic Modelling':
        st.header('Topic Modelling')

        form = st.form(key='topic_modelling_form')
        uploaded_file = form.file_uploader('Upload a file in .csv, .txt, .doc, .docx, or .pdf format')

        custom_parameters = form.text_input('Add Custom Parameters')
        insight = form.text_input('Tell us a little bit about your data')

        submit_button = form.form_submit_button(label='Analyse')

        if submit_button and uploaded_file is not None:
            data = topic_modelling(uploaded_file, custom_parameters, insight)

            st.subheader('Results')

            # Display the summary
            st.write("## Summary")
            st.write(data["summary"])

            try:
                st.write("## Topics")
                topic_table = "<table><tr><th>Topic</th></tr>"
                for topic in data["topics"]:
                    topic_table += f"<tr><td>{topic}</td></tr>"
                topic_table += "</table>"
                st.write(topic_table, unsafe_allow_html=True)
            except Exception as e:
                pass

            try:
                # Create a table for the types
                st.write("## Types")
                type_table = "<table><tr><th>Type</th><th>Value</th></tr>"
                for t, values in data["types"].items():
                    for value in values:
                        type_table += f"<tr><td>{t}</td><td>{value}</td></tr>"
                type_table += "</table>"
                st.write(type_table, unsafe_allow_html=True)
            except Exception: 
                pass

            try:
                # Plot the distribution graph
                fig = go.Figure()
                labels = [t["label"] for t in data["topic_distribution"]]
                values = [t["value"][0] for t in data["topic_distribution"]]
                fig.add_trace(go.Bar(x=labels, y=values))
                fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 0.3], xaxis_title="Topic", yaxis_title="Value")
                st.plotly_chart(fig)
            except Exception: 
                pass
            
            try:
                # Display the topic_keywords dict
                st.write("## Topic Keywords")
                for tk in data["topic_keywords"]:
                    st.write(f"### {tk['topic']}")
                    st.write(", ".join(tk["keywords"]))
            except Exception:
                pass
            
            try:
                # Create a word cloud
                st.write("## Word Cloud")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                wordcloud = WordCloud(width=800, height=400).generate(data["word_cloud"])
                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot()
            except Exception:
                pass

            # Display the custom parameters
            st.write("## Custom Parameters")
            st.write(data["custom_parameters"])
 

    elif analysis_choice == 'Entity Recognition':
        st.header('Entity Recognition')

        form = st.form(key='entity_recognition_form')
        uploaded_file = form.file_uploader('Upload a file in .csv, .txt, .doc, .docx, or .pdf format')

        custom_parameters = form.text_input('Add Custom Parameters')
        insight = form.text_input('Tell us a little bit about your data')

        submit_button = form.form_submit_button(label='Analyse')

        if submit_button and uploaded_file is not None:
            data = entity_recognition(uploaded_file, custom_parameters, insight)

            df = data

            # Display the summary
            st.write("## Summary")
            st.write(data["summary"])

            # Named entities table
            st.write("## Named Entities")
            st.table(df["named_entities"])

            # List of entity types pie chart
            st.write("## Entity Types")
            fig = px.pie(df["list_of_entity types"], names=df["list_of_entity types"], title="List of Entity Types")
            st.plotly_chart(fig)

            # Contextual info table
            st.write("## Contextual Info")
            context_df = pd.DataFrame(df["contextual_info"])
            st.table(context_df)

            # Entity occurrences graph
            st.write("## Entity Occurrences")
            fig = px.bar(df["entity_occurrences"], x="entity", y="count", title="Entity Occurrences")
            st.plotly_chart(fig)


            # Display the summary
            st.write("## Custom Q&A")
            st.write(data["custom_parameters"])

    elif analysis_choice == 'Actionable Insights':
        st.header('Actionable Insights')

        form = st.form(key='actionable_insights_form')
        uploaded_file = form.file_uploader('Upload a file in .csv, .txt, .doc, .docx, or .pdf format')

        custom_parameters = form.text_input('Add Custom Parameters')
        insight = form.text_input('Tell us a little bit about your data')

        submit_button = form.form_submit_button(label='Analyse')

        if submit_button and uploaded_file is not None:
            data = actionable_insights(uploaded_file, custom_parameters, insight)

            # Display the summary and custom parameters
            st.write("Summary: " + data["summary"])
            st.write("Custom parameters: " + data["custom_parameters"])

            # Create a two-column table for the common feedback
            common_feedback = pd.DataFrame(columns=["Type", "Feedback"])
            for feedback_type, feedback_list in data.items():
                if feedback_type.startswith("common"):
                    for feedback in feedback_list:
                        common_feedback = common_feedback.append({
                            "Type": feedback_type,
                            "Feedback": list(feedback.values())[0]
                        }, ignore_index=True)
            st.write("Common Feedback:")
            st.table(common_feedback)
                
            # Create a table for the actionable insights
            insights = pd.DataFrame(data["actionable_insights"])
            st.write("Actionable Insights:")
            st.table(insights)


if __name__ == '__main__':
    main()