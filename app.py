# =================================================================================
#                 Türkçe Ürün Yorumlarına Dayalı RAG Chatbot
# =================================================================================
# Bu Streamlit uygulaması, FAISS ve Google Gemini kullanarak ürün yorumlarına
# dayalı soruları yanıtlayan bir chatbot sunar.

# --- Gerekli Kütüphanelerin İçe Aktarılması ---
import streamlit as st                 # Web arayüzünü oluşturmak için (UI Framework)
import google.generativeai as genai    # Google Gemini modelini kullanmak için (LLM API)
import os                              # Ortam değişkenlerini okumak için (API Anahtarı)
import faiss                           # Vektör veritabanı oluşturma ve arama için (Vector DB)
import pickle                          # Python nesnelerini (meta veri listesi) kaydetmek/yüklemek için
import numpy as np                     # Sayısal işlemler ve diziler için (FAISS için gerekli)
from sentence_transformers import SentenceTransformer # Metinleri vektörlere dönüştürmek için (Embedding Modeli)

# --- Adım 1: Yapılandırma ve Kaynak Yükleme ---

# @st.cache_resource: Bu fonksiyonların sonuçları cache'lenir, 
# böylece uygulama her yenilendiğinde tekrar çalıştırılmazlar. Bu, hız ve verimlilik sağlar.
@st.cache_resource 
def load_api_key_and_configure_gemini():
    """Google API Anahtarını HF Secrets'tan (ortam değişkeni) yükler ve Gemini modelini yapılandırır."""
    try:
        # 1. API Anahtarını Oku: Hugging Face Secrets'a eklenen anahtarı ortam değişkeni olarak okur.
        api_key = os.environ.get('GOOGLE_API_KEY') 
        if api_key is None:
            st.error("HATA: Google API Anahtarı Hugging Face Secrets'ta bulunamadı. Lütfen Space > Settings > Secrets bölümünü kontrol edin.")
            st.stop()
            
        # 2. Gemini Kütüphanesini Yapılandır: Alınan API anahtarı ile google.generativeai kütüphanesini ayarlar.
        genai.configure(api_key=api_key)
        
        # 3. Gemini Modelini Yükle: Kullanılacak LLM modelini (örn: 'gemini-flash-latest') hazırlar.
        model = genai.GenerativeModel('gemini-flash-latest')
        
        st.success("✅ Google API Anahtarı yüklendi ve Gemini modeli yapılandırıldı.")
        # 4. Modeli Döndür: Yapılandırılmış modeli ana uygulamada kullanmak üzere döndürür.
        return model
    except Exception as e:
        st.error(f"HATA: API Anahtarı veya Gemini yapılandırması başarısız: {e}")
        st.stop()

@st.cache_resource
def load_retrieval_system():
    """Embedding modelini, FAISS indeksini ve meta veriyi diskten yükler."""
    try:
        st.info("⏳ Arama sistemi (Embedding Modeli, FAISS, Meta Veri) yükleniyor...")
        
        # 1. Embedding Modelini Yükle: Metinleri ve sorguları vektöre çevirecek modeli yükler.
        embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # 2. FAISS İndeksini Yükle: Önceden hesaplanmış vektörleri içeren veritabanını yükler.
        index = faiss.read_index("vektor_indeksi.faiss")
        
        # 3. Meta Veriyi Yükle: Hangi vektörün hangi yoruma ait olduğunu belirten listeyi yükler.
        with open("meta_veri.pkl", 'rb') as f:
            processed_data = pickle.load(f)
            
        st.success("✅ Arama sistemi başarıyla yüklendi.")
        # 4. Yüklenen Bileşenleri Döndür: Modeli, indeksi ve meta veriyi ana uygulamada kullanmak üzere döndürür.
        return embedding_model, index, processed_data
    except FileNotFoundError:
        st.error("HATA: 'vektor_indeksi.faiss' veya 'meta_veri.pkl' dosyaları bulunamadı! Lütfen bu dosyaların Space'te mevcut olduğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"HATA: Arama sistemi yüklenirken hata: {e}")
        st.stop()

# --- Adım 2: RAG Pipeline Fonksiyonları ---

# Global değişkenler (load_retrieval_system tarafından doldurulacak)
embedding_model = None
index = None
processed_data = None

def retrieve_contexts(query, k=10): 
    """Verilen sorgu için FAISS veritabanından en ilgili 'k' metin parçasını (chunk) bulur."""
    if embedding_model is None or index is None or processed_data is None:
        return [] 
        
    query_vector = embedding_model.encode([query]) 
    try:
        distances, indices = index.search(np.array(query_vector).astype('float32'), k) 
        contexts = [processed_data[idx]['text'] for idx in indices[0]] 
        return contexts
    except Exception as e:
        st.error(f"FAISS araması sırasında hata: {e}")
        return []

def answer_with_rag(query, generation_model_from_cache):
    """Retrieval (Arama) ve Generation (Üretim) adımlarını birleştirir."""
    
    # 1. Retrieval: İlgili yorum parçalarını bul
    relevant_contexts = retrieve_contexts(query)
    if not relevant_contexts:
        return "Elimdeki yorumlarda bu konuyla ilgili bilgi bulmakta zorlanıyorum."
        
    context_str = "\n\n---\n\n".join(relevant_contexts)
    
    # 2. Generation: Gemini için prompt'u oluştur
    prompt = f"""Sen, Türkçe ürün yorumlarına dayanarak ürünler hakkında bilgi veren ve tavsiyelerde bulunan bir asistansın. 
Görevin, aşağıda sağlanan ürün yorumu parçalarını (Bağlam) dikkatlice analiz etmek ve sorulan soruya en mantıklı ve genel eğilimi yansıtan cevabı oluşturmaktır.
Eğer bağlamda sorunun cevabı doğrudan geçmiyorsa, verilen bilgileri yorumlayarak, çıkarım yaparak veya genel bir özet sunarak soruyu yanıtla.
Cevabını her zaman kibar ve yardımcı bir tonda tut.

BAĞLAM:
{context_str}

SORU:
{query}

CEVAP:"""
    
    # Gemini'den cevap üretmesini iste
    try:
        response = generation_model_from_cache.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7) 
        )
        # Güvenlik filtrelerini kontrol et
        if not response.parts:
             return "⚠️ Üretilen cevap güvenlik filtrelerine takıldı. Lütfen sorunuzu değiştirin."
        return response.text
    except Exception as e:
        st.error(f"HATA: Gemini'den cevap üretilirken sorun oluştu: {e}")
        return "Cevap üretirken bir hatayla karşılaştım."

# --- Adım 3: Streamlit Arayüzünü Oluşturma ve Çalıştırma ---

def main():
    """Streamlit arayüzünü oluşturur ve chatbot mantığını çalıştırır."""
    
    st.set_page_config(page_title="Ürün Yorum Chatbot'u", page_icon="📦")
    st.title("📦 Türkçe Ürün Yorum Chatbot'u")
    st.caption("Veri Seti: Turkish Product Reviews (Alt Küme) | Model: Gemini & Sentence Transformers")

    # Uygulama başlarken gerekli modelleri ve veriyi yükle
    generation_model = load_api_key_and_configure_gemini()
    global embedding_model, index, processed_data 
    embedding_model, index, processed_data = load_retrieval_system()

    # Sohbet geçmişini tut
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları yazdır
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan yeni girdi al
    if user_prompt := st.chat_input("Bir ürün veya özellik hakkında soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Chatbot'un cevabını al ve göster
        with st.chat_message("assistant"):
            with st.spinner("Yorumlar taranıyor ve cevap oluşturuluyor..."):
                response = answer_with_rag(user_prompt, generation_model)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Python script'i doğrudan çalıştırıldığında main() fonksiyonunu çağır
if __name__ == "__main__":
    main()
