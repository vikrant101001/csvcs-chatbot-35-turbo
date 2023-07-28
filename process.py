from pathlib import Path
import sys
import faiss
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
import requests
import os


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


def train():

  # Check there is data fetched from the database

  trainingData = list(Path("training/facts/").glob("**/*.*"))

  # Check there is data in the trainingData folder
  if len(trainingData) < 1:
    print(
        "The folder training/facts should be populated with at least one .txt or .md file.",
        file=sys.stderr)
    return

  data = []
  for training in trainingData:
    with open(training) as f:
      print(f"Add {f.name} to dataset")
      data.append(f.read())

  textSplitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                chunk_overlap=0)

  docs = []
  for sets in data:
    docs.extend(textSplitter.split_text(sets))
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
  store = FAISS.from_texts(docs, embeddings)

  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("faiss.pkl", "wb") as f:
    pickle.dump(store, f)


def runPrompt():
  index = faiss.read_index("training.index")

  with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  with open("training/master.txt", "r") as f:
    promptTemplate = f.read()

  prompt = Prompt(template=promptTemplate,
                  input_variables=["history", "context", "question"])

  llmChain = LLMChain(prompt=prompt,
                      llm=OpenAIChat(
                          temperature=0.5,
                          model_name='gpt-3.5-turbo',
                          openai_api_key=os.environ["OPENAI_API_KEY"]))

  def onMessage(question, history):
    # Check if the question is related to the user's location
    if "need" in question.lower():
      location = input(
          "Please provide your complete location so that we can find the nearest required professional for you: "
      )
      latitude, longitude = geocode(location, os.environ["MAP_KEY"])
      # Store the latitude and longitude in your database
      # Perform actions related to address-based functionality
      # Sort professionals based on proximity using latitude and longitude

    docs = store.similarity_search(question)
    contexts = []
    for i, doc in enumerate(docs):
      contexts.append(f"Context {i}:\n{doc.page_content}")

    answer = llmChain.predict(question=question,
                              context="\n\n".join(contexts),
                              history=history)

    return answer

  history = []
  while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    print(f"Bot: {answer}")
    history.append(f"Human: {question}")
    history.append(f"Bot: {answer}")


# Example usage
train()
runPrompt()
