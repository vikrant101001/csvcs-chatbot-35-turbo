from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import requests

pointer = 0


def geocode(address, access_token):
  if not address:
    return None, None

  url = f'https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={access_token}'
  response = requests.get(url)
  data = response.json()
  if data['features']:
    longitude, latitude = data['features'][0]['center']
    return latitude, longitude
  else:
    return None, None


index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
  store = pickle.load(f)

store.index = index

with open("training/master.txt", "r") as f:
  promptTemplate = f.read()

prompt = Prompt(template=promptTemplate,
                input_variables=["history", "context", "question"])

app = Flask(__name__)
cors = CORS(app)


@app.route("/", methods=["GET"])
def index():
  return "API Online"


@app.route("/", methods=["POST"])
def ask():

  reqData = request.get_json()
  if reqData['secret'] == os.environ["API_SECRET"]:
    print("secret matches")
    try:
      # Retrieve the user's question from the request
      user_question = reqData['question']
      if pointer == 1:
        location = user_question
        latitude, longitude = geocode(
          location,
          'pk.eyJ1IjoiZXZ2YWhlYWx0aCIsImEiOiJjbGp5anJjY2IwNGlnM2RwYmtzNGR0aGduIn0.Nx4jv-saalq2sdw9qKuvbQ'
        )
      question = user_question

      user_history = reqData['history']
      history = user_history
      # Rest of the code...
      llmChain = LLMChain(prompt=prompt,
                          llm=OpenAIChat(
                            temperature=0.5,
                            model_name="gpt-3.5-turbo",                            openai_api_key=os.environ["OPENAI_API_KEY"]))

      def onMessage(question, history):
        pointer = 0
        if "live" in question.lower() or "need" in question.lower():
          pointer = 1
          answer = "Please provide your complete location so that we can find the nearest required professional for you: "

        docs = store.similarity_search(question)
        contexts = []
        for i, doc in enumerate(docs):
          contexts.append(f"Context {i}:\n{doc.page_content}")
          if pointer == 0:
            answer = llmChain.predict(question=question,                                      context="\n\n".join(contexts),
                                      history=history)
        return answer

      return jsonify({
        "answer":
        onMessage(reqData['question'], reqData['history']),
        "success":
        True
      })

    except:

      return jsonify({
        "answer": None,
        "success": False,
        "message": "Error"
      }), 400

  else:
    print("secret doesnt match")
    return jsonify({
      "answer": None,
      "success": False,
      "message": "Unauthorised"
    })


app.run(host="0.0.0.0", port=3000)
