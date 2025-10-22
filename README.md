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
