# Bu komut, aşağıdaki GÜNCEL kodun tamamını 'app.py' adlı bir dosyaya yazar.

import streamlit as st
import google.generativeai as genai
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
# Secrets yerine Ortam Değişkeni okumak için os GEREKLİ
from google.colab import userdata 

# --- ÖN YÜKLEME FONKSİYONLARI ---
@st.cache_resource
def load_api_key_and_configure_gemini():
    """Google API Anahtarını ORTAM DEĞİŞKENİNDEN yükler ve Gemini'yi yapılandırır."""
    try:
        api_key = os.environ.get('GOOGLE_API_KEY_STREAMLIT')
        if api_key is None:
            st.error("HATA: Google API Anahtarı ortam değişkeninde bulunamadı.")
            st.stop()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        st.success("✅ Google API Anahtarı yüklendi ve Gemini modeli yapılandırıldı.")
        return model
    except Exception as e:
        st.error(f"HATA: API Anahtarı yüklenirken/yapılandırılırken sorun oluştu: {e}")
        st.stop()

@st.cache_resource
def load_retrieval_system():
    """Embedding modelini, FAISS indeksini ve meta veriyi yükler."""
    try:
        st.info("⏳ Arama sistemi (Embedding Modeli, FAISS, Meta Veri) yükleniyor...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        index = faiss.read_index("vektor_indeksi.faiss")
        with open("meta_veri.pkl", 'rb') as f:
            processed_data = pickle.load(f)
        st.success("✅ Arama sistemi başarıyla yüklendi.")
        return embedding_model, index, processed_data
    except FileNotFoundError:
        st.error("HATA: 'vektor_indeksi.faiss' veya 'meta_veri.pkl' dosyaları bulunamadı!")
        st.stop()
    except Exception as e:
        st.error(f"HATA: Arama sistemi yüklenirken hata: {e}")
        st.stop()

# --- RAG Fonksiyonları ---
embedding_model = None
index = None
processed_data = None

# --- k=10 arama ---
def retrieve_contexts(query, k=10): 
    if embedding_model is None or index is None or processed_data is None:
        st.error("Arama sistemi henüz yüklenmedi.")
        return []
    query_vector = embedding_model.encode([query])
    try:
        D, I = index.search(np.array(query_vector).astype('float32'), k)
        contexts = [processed_data[idx]['text'] for idx in I[0]]
        return contexts
    except Exception as e:
        st.error(f"FAISS araması sırasında hata: {e}")
        return []

def answer_with_rag(query, generation_model_from_cache):
    relevant_contexts = retrieve_contexts(query)
    if not relevant_contexts:
        return "Elimdeki yorumlarda bu konuyla ilgili bilgi bulmakta zorlanıyorum."
    context_str = "\n\n---\n\n".join(relevant_contexts)
    
    # --- Esnek Prompt Kullanılıyor ---
    prompt = f"""Sen, Türkçe ürün yorumlarına dayanarak ürünler hakkında bilgi veren ve tavsiyelerde bulunan bir asistansın. 
Görevin, aşağıda sağlanan ürün yorumu parçalarını (Bağlam) dikkatlice analiz etmek ve sorulan soruya en mantıklı ve genel eğilimi yansıtan cevabı oluşturmaktır.
Eğer bağlamda sorunun cevabı doğrudan geçmiyorsa, verilen bilgileri yorumlayarak, çıkarım yaparak veya genel bir özet sunarak soruyu yanıtla.
Cevabını her zaman kibar ve yardımcı bir tonda tut.

BAĞLAM:
{context_str}

SORU:
{query}

CEVAP:"""
    # --- ---

    try:
        # --- Temperature Ayarı ---
        response = generation_model_from_cache.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7 
            )
        )
        if not response.parts:
             return "⚠️ Üretilen cevap güvenlik filtrelerine takıldı. Lütfen sorunuzu değiştirin."
        return response.text
    except Exception as e:
        st.error(f"HATA: Gemini'den cevap üretilirken sorun oluştu: {e}")
        return "Cevap üretirken bir hatayla karşılaştım."

# --- STREAMLIT UYGULAMA ARAYÜZÜ ---
st.set_page_config(page_title="Ürün Yorum Chatbot'u", page_icon="📦")
st.title("📦 Türkçe Ürün Yorum Chatbot'u")
st.caption("Veri Seti: Turkish Product Reviews (Alt Küme) | Model: Gemini & Sentence Transformers")

generation_model = load_api_key_and_configure_gemini()
embedding_model, index, processed_data = load_retrieval_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Bir ürün veya özellik hakkında soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yorumlar taranıyor ve cevap oluşturuluyor..."):
            response = answer_with_rag(user_prompt, generation_model)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
