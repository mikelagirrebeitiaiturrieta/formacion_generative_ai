"""
author: Mikel Agirrebeitia
"""

import os
from dotenv import load_dotenv

import streamlit as st
from langchain.prompts import PromptTemplate
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from create_db import VectorDB


