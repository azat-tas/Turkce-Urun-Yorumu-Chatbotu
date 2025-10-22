# Türkçe Ürün Yorumlarına Dayalı RAG Chatbotu

Retrieval Augmented Generation (RAG) tekniği kullanarak Türkçe ürün yorumlarına dayalı soruları yanıtlayan ve tavsiyelerde bulunan bir chatbot.

## Projenin Amacı

Bu chatbot, kullanıcılardan gelen ürünlerle ilgili soruları (örneğin, bir ürün hakkındaki genel görüşler, belirli bir özelliğin performansı) anlamayı ve Hugging Face'deki geniş bir Türkçe ürün yorumu veri setinden ilgili bilgileri bularak yanıtlar üretmeyi hedefler. Temel amaç, kullanıcılara ürün araştırmalarında yorumlara dayalı özet bilgiler sunmaktır.

## Veri Seti Hakkında Bilgi

* **Veri Seti:** `turkish_product_reviews` (Hugging Face Datasets).
* **Kaynak:** Fatih Barmanbay tarafından derlenmiş, çeşitli e-ticaret sitelerinden toplanmış Türkçe ürün yorumları.
* **Boyut ve Kullanım:** Orijinal veri seti 235.000'den fazla yorum içermektedir. Bu projede, Colab ortamındaki kaynak kısıtlamaları ve işlem süresini (özellikle Embedding adımı) yönetilebilir kılmak amacıyla, veri setinden rastgele seçilmiş **20.000 yorumluk dengeli bir alt küme** (yaklaşık 10.000 olumlu, 10.000 olumsuz yorum) kullanılmıştır. Bu yaklaşım, chatbot'un hem olumlu hem de olumsuz görüşler hakkında bilgi sahibi olmasını sağlar.
* **Özellikler:** Kullanılan temel sütunlar `sentence` (yorum metni) ve `sentiment` (duygu etiketi: 0=olumsuz, 1=olumlu).

## Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, RAG (Retrieval Augmented Generation) mimarisine dayanmaktadır:

1.  **Retrieval (Bilgi Erişimi):**
    * **Chunking:** Yorum metinleri (`sentence`), anlamsal bütünlüğü korumak amacıyla ~250 kelimelik, 50 kelime örtüşmeli parçalara (chunk) ayrılmıştır.
    * **Embedding:** Metin parçaları, Türkçe'yi de destekleyen `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers) modeli kullanılarak 768 boyutlu vektörlere dönüştürülmüştür.
    * **Vektör Veritabanı:** Oluşturulan vektörler, hızlı ve verimli anlamsal arama yapabilmek için `FAISS` (Facebook AI Similarity Search) kütüphanesi (IndexFlatL2 indeksi) kullanılarak bir vektör indeksine kaydedilmiştir. Her vektöre karşılık gelen meta veriler (puan, yorum önizlemesi) ayrı bir `.pkl` dosyasında saklanmıştır.
2.  **Generation (Yanıt Üretimi):**
    * Kullanıcı sorgusu geldiğinde, önce embedding model ile vektöre çevrilir.
    * FAISS kullanılarak bu sorgu vektörüne en yakın `k=10` adet yorum parçası (chunk) bulunur.
    * Bu bulunan yorum parçaları (bağlam olarak) ve kullanıcının orijinal sorusu, Google'ın `gemini-flash-latest` modeline özel bir prompt ile gönderilir.
    * Prompt, modelin cevabını sağlanan bağlama dayandırmasını, çıkarım yapmasını ve yorumlardaki genel eğilimi özetlemesini ister. `temperature=0.7` ayarı kullanılarak cevapların dengeli bir yaratıcılıkta olması hedeflenmiştir.
3.  **Web Arayüzü:**
    * Uygulama, `Streamlit` kütüphanesi kullanılarak geliştirilmiş interaktif bir sohbet arayüzü sunar.
    * Uygulama, Hugging Face Spaces platformunda yayınlanmıştır.

## Elde Edilen Sonuçlar ve Örnek Sorular

Geliştirilen RAG chatbot'u, kullanılan 20.000 ürün yorumu kapsamında çeşitli sorulara yanıt verebilmektedir. Arama (Retrieval) adımı, sorguyla ilgili yorum parçalarını bulmakta, Üretim (Generation) adımı ise bu parçaları kullanarak tutarlı ve bağlama dayalı cevaplar oluşturmaktadır. Kullanılan alt küme nedeniyle, veri setinde bulunmayan ürünler veya konular hakkında "Elimdeki yorumlara göre bu konuda bir şey söyleyemem." veya benzeri yanıtlar vermesi normaldir.

**Chatbot'a Sorabileceğiniz Örnek Sorular:**

* Kablosuz kulaklıklar hakkında genel olarak ne düşünülüyor?
* Telefonların kamera kalitesi hakkında yorumlarda neler söyleniyor?
* Oyun oynamak için klavye arıyorum, yorumlarda önerilen var mı?
* Akıllı saatlerin en çok şikayet edilen özellikleri neler?
* X marka (popüler bir marka) ürünleri hakkında yorumlar nasıl?
* İyi bir oyuncu koltuğu önerir misin?

*(Not: Chatbot'un cevapları, veri setindeki mevcut yorumlara ve Gemini modelinin yorumlama yeteneğine göre değişiklik gösterebilir.)*

## Web Linki

Chatbot'un çalışan web arayüzüne aşağıdaki linkten erişebilirsiniz:

**https://huggingface.co/spaces/MuhammetAzat/Urun-Yorum-Chatbotu-M**


## Çalıştırma Kılavuzu

Bu bölüm, projeyi (örneğin Google Colab gibi bir ortamda veya kendi bilgisayarınızda) çalıştırmak için izlenmesi gereken adımları özetlemektedir.

### Gereksinimler

* Python (3.9+)
* Git (Proje dosyalarını indirmek için)
* Google API Anahtarı ([Google AI Studio](https://ai.google.dev/) üzerinden alınabilir)

### Adımlar

1.  **Projeyi Klonlama (Neden Gerekli?):**
    Projenin çalışması için gereken tüm kod (`app.py`), bağımlılık listesi (`requirements.txt`) ve veri dosyaları (`.faiss`, `.pkl`) GitHub üzerinde barındırılmaktadır. Bu dosyaları kendi çalışma ortamınıza (bilgisayarınıza veya Colab'a) indirmenin standart yolu `git clone` komutunu kullanmaktır.
    ```bash
    git clone [https://github.com/azat-tas/Turkce-Urun-Yorumu-Chatbotu.git](https://github.com/azat-tas/Turkce-Urun-Yorumu-Chatbotu.git)
    cd Turkce-Urun-Yorumu-Chatbotu 
    ```
    *(Not: `vektor_indeksi.faiss` gibi büyük dosyaların doğru indirilmesi için sisteminizde Git LFS'in kurulu olması gerekebilir: `git lfs install`)*

2.  **(Önerilen) Sanal Ortam Oluşturma:**
    Projenin kütüphane bağımlılıklarının sisteminizdeki diğer projelerle çakışmasını önlemek için bir sanal ortam (virtual environment) oluşturup etkinleştirmeniz **şiddetle tavsiye edilir**. Proje klasörünün içindeyken:
    ```bash
    # Sanal ortamı oluştur (venv adıyla)
    python -m venv venv 
    # Sanal ortamı etkinleştir
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Kütüphaneleri Yükleme:**
    Projenin ihtiyaç duyduğu tüm Python kütüphanelerini `requirements.txt` dosyasını  kullanarak otomatik olarak yükleyin:
    ```bash
    pip install -r requirements.txt 
    ```

4.  **Google API Anahtarını Ayarlama:**
    Uygulamanın Google Gemini modelini kullanabilmesi için API anahtarınızı ayarlamanız gerekir. Uygulama (`app.py`), anahtarı `GOOGLE_API_KEY_STREAMLIT` adlı bir ortam değişkeninden okur.
    * **Colab Kullanımı:** Sol menüdeki Anahtar (🔑) simgesine tıklayın. `GOOGLE_API_KEY` adıyla yeni bir "Secret" oluşturun, anahtarınızı yapıştırın ve "Notebook access" iznini açın. (Uygulamayı başlatan kod bu Secret'ı okuyacaktır).
    * **Yerel Kullanım:** Uygulamayı çalıştırmadan önce terminalinizde anahtarınızı ortam değişkeni olarak ayarlayın (işletim sisteminize uygun komutu kullanın, örn. Mac/Linux: `export GOOGLE_API_KEY_STREAMLIT='YOUR_API_KEY'`).

5.  **Uygulamayı Başlatma:**
    Kurulum tamamlandıktan sonra, projenin ana klasöründeyken aşağıdaki komutu  çalıştırın:
    ```bash
    streamlit run app.py 
    ```
    Bu komut, chatbot arayüzünü tarayıcınızda açacaktır (genellikle `http://localhost:8501`).




    ## Kullanım Kılavuzu (Product Guide)

Bu kılavuz, yayınlanan chatbot web arayüzünün nasıl kullanılacağını açıklar.

### Erişim

Chatbot'a aşağıdaki Hugging Face Space linki üzerinden erişilebilir:

**https://huggingface.co/spaces/MuhammetAzat/Urun-Yorum-Chatbotu-M**


### Arayüz Açıklaması

Linke tıkladığınızda basit bir sohbet arayüzü ile karşılaşacaksınız:

1.  **Başlık:** "📦 Türkçe Ürün Yorum Chatbot'u".
2.  **Alt Başlık:** Kullanılan veri seti ve modeller hakkında kısa bilgi.
3.  **Sohbet Alanı:** Daha önceki konuşmalarınız (varsa) burada görüntülenir.
4.  **Giriş Kutusu:** Sayfanın en altında "Bir ürün veya özellik hakkında soru sorun..." yazan bir metin kutusu bulunur. Sorularınızı buraya yazıp Enter'a basarak chatbot ile etkileşime geçebilirsiniz.
5.  **Durum Mesajları:** Chatbot cevap vermeden önce "Yorumlar taranıyor ve cevap oluşturuluyor..." gibi durum mesajları görünebilir.

*(İsteğe Bağlı: Buraya arayüzün basit bir ekran görüntüsünü ekleyebilirsiniz.)*

### Chatbot'u Test Etme ve Yetenekleri

Chatbot, kendisine öğretilen 20.000 Türkçe ürün yorumuna dayanarak soruları yanıtlamaya çalışır. Aşağıdaki türde sorular sorarak yeteneklerini test edebilirsiniz:

* **Genel Ürün Kategorisi Hakkında:**
    * `Kablosuz kulaklıklar hakkında genel olarak ne düşünülüyor?`
    * `Akıllı saatlerin avantajları ve dezavantajları nelerdir?`
* **Belirli Bir Özellik Hakkında:**
    * `Telefonların kamera kalitesi hakkında yorumlarda neler söyleniyor?`
    * `Laptopların pil ömrü nasıl genelde?`
* **Kullanım Amacına Yönelik Tavsiye:**
    * `Oyun oynamak için klavye arıyorum, yorumlarda önerilen var mı?`
    * `Sessiz çalışan bir mouse önerir misin?`
* **Olumlu/Olumsuz Yönler:**
    * `Robot süpürgelerin en çok şikayet edilen özellikleri neler?`
    * `Monitör alırken nelere dikkat etmek gerektiği yorumlarda geçiyor mu?`
* **Marka/Model (Popülerse):**
    * `X marka (popüler bir marka) telefonlar hakkında yorumlar nasıl?`
    * `Y modeli (bilinen bir model) hakkında olumsuz yorum var mı?`

**Beklenen Davranış:**

* Chatbot, cevabını veri setindeki yorumlara dayandırmaya çalışacaktır.
* Eğer sorulan konuyla ilgili yeterli veya alakalı yorum bulamazsa, "Elimdeki yorumlara göre bu konuda bir şey söyleyemem." veya benzeri bir yanıt verecektir. Bu, chatbot'un halüsinasyon üretmediğini gösterir.
* Cevaplar, Gemini modelinin yorumlama yeteneğine göre değişiklik gösterebilir.
