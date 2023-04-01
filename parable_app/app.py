import os
import openai
import json
from flask import Flask, jsonify, redirect, render_template, request, url_for
import pandas as pd
import PyPDF2
import docx2txt
from flask_cors import CORS, cross_origin


openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def extract_pdf_text(file):
    # Read PDF file and extract text
    text = ''
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_doc_text(file):
    # Read Word document file and extract text
    text = docx2txt.process(file)
    return text

@app.route('/sentiment-analysis', methods=('GET','POST'))
def index():
    if request.method == "POST":

        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(request.files['file'])
        elif file.filename.endswith('.pdf'):
            data = extract_pdf_text(file)
        elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
            data = extract_doc_text(file)
        custom_para = request.form['custom_parameters']
        insight = request.form['insight']

        prompt = f'''
        You are a text to insight service. Perform sentiment analysis on the following. Also consider the custom parameters field:
        data: {data},
        custom parameters: {custom_para},
        insight: {insight},
        '''
        output = '''
        Your output should be in the JSON format: 
                {
        "positive_words": [
            "list of positive phrases"
        ],
        "negative_words": [
            "list of negative phrases"
        ],
        "neutral_words": [
            "list of neutral phrases"
        ],
        "custom_parameters": "answer to the custom parameters",
        "summary": "summary goes here"
        }

        '''
        prompt = prompt + output
        prompt = [{"role": "user", "content":prompt}]
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0.6,
                )
        response = response.choices[0].message.content

        return jsonify(response)

    response = request.args.get("output")

    return jsonify(response)
    
@app.route('/entity_recognition', methods=('GET','POST'))
def entity_recognition():
    if request.method == "POST":

        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(request.files['file'])
        elif file.filename.endswith('.pdf'):
            data = extract_pdf_text(file)
        elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
            data = extract_doc_text(file)
        custom_para = request.form['custom_parameters']
        insight = request.form['insight']
        prompt = f'''You are a text to insight service. Perform entity recognition on the following. 
        Also consider the custom parameters field. The insight tells a little about the data. 
        Return list of named entities, list of entity types, contextual info (The context in which each named entity appears in the text, such as the surrounding words, sentences, or paragraphs.) and number of entity occurences.
        data: {data},
        custom parameters: {custom_para},
        insight: {insight},
        ''' 
        output = '''
        Return the output in the following JSON format:

        {
            "named_entities": [
                "list of named entities"
            ],
            "list_of_entity types": [],
            "contextual_info": [
                {"entity": "entity", "context": "context"},
                "..."
            ],
            "entity_occurrences": [
                {"entity": "entity", "count": "count of entity occurrences"},
                "..."
            ],
            "custom_parameters": "answer to the custom parameters",
            "summary": "summary goes here"
        }
        '''
        prompt = prompt + output
        prompt = [{"role": "user", "content":prompt}]
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0.6,
                )
        response = response.choices[0].message.content

        return jsonify(response)

    response = request.args.get("output")

    return jsonify(response)

@app.route('/topic-modelling', methods=('GET','POST'))
def topic_modelling():
    if request.method == "POST":
        
        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(request.files['file'])
        elif file.filename.endswith('.pdf'):
            data = extract_pdf_text(file)
        elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
            data = extract_doc_text(file)
        custom_para = request.form['custom_parameters']
        insight = request.form['insight']
        prompt = f'''You are a text to insight service. Perform topic modelling on the following. 
        Also consider the custom parameters field. Extract important phrases and analyse the type of phrase.
        Types include feature suggestions, product improvements, suggestions, critique etc. Also return the 
        topic distribution, topic keywords (list of keywords associated with each topic), topic hierarchy (how topics are related to each other) and word cloud (list of common occuring words):
        data: {data},
        custom parameters: {custom_para},
        insight: {insight}'''

        output = '''
        Return the output in the following JSON format:

        {
            "topics": [
                "list of topics"
            ],
            "types": {
                "type1": ["phrase"],
                "type2": ["phrase"],
                "..."
            },
            "topic_distribution": [
                {"label": "topic label", "value": "[list of topic distribution values]"}
            ],
            "topic_keywords": [
                {"topic": "topic label", "keywords": "[list of keywords associated with the topic]"},
                "..."
            ],
            "topic_hierarchy": "[topic hierarchy goes here]",
            "word_cloud": "[word cloud goes here]",
            "custom_parameters": "answer to the custom parameters",
            "summary": "summary goes here"
        }
        '''
        prompt = prompt + output

        # Send prompt to OpenAI and get output 
        prompt = [{"role": "user", "content":prompt}]
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0.6,
                )
        response = response.choices[0].message.content

        return jsonify(response)

    response = request.args.get("output")

    return jsonify(response)



@app.route('/trend-analysis', methods=['POST'])
def actionable_insights():
    if request.method == "POST":

        file = request.files['file']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(request.files['file'])
        elif file.filename.endswith('.pdf'):
            data = extract_pdf_text(file)
        elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
            data = extract_doc_text(file)
        custom_para = request.form['custom_parameters']
        insight = request.form['insight']

        prompt = f'''
        You are a text to insight service. Peform analysis and retrieve actionable insights from the following. Also consider the custom parameters field:
        data: {data},
        custom parameters: {custom_para},
        insight: {insight},
        '''
        output = '''
        Your output should be in the JSON format: 
                {
        "actionable_insights": [
            "list of actionable insights"
        ],
        "custom_parameters": "answer to the custom parameters",
        "summary": "summary goes here"
        }

        '''
        prompt = prompt + output
        prompt = [{"role": "user", "content":prompt}]
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0.6,
                )
        response = response.choices[0].message.content

        return jsonify(response)

    response = request.args.get("output")

    return jsonify(response)


    
   
