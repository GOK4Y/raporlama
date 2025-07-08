import os
import io
import urllib.parse
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import google.generativeai as genai
from bs4 import BeautifulSoup
from weasyprint import HTML
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# --- FastAPI Uygulama Başlatma ---
app = FastAPI(
    title="Mülakat Raporu Oluşturucu API",
    description="CSV verilerinden tutarlı ve görsel olarak zenginleştirilmiş PDF mülakat raporları oluşturur.",
    version="1.4.0",  # Sürüm, PDF oluşturma kütüphanesi WeasyPrint olarak güncellendi
)

# --- Gemini API Yapılandırması ---
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
    )
except KeyError:
    raise RuntimeError(
        "GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasında ayarlayın."
    )

# --- Yardımcı Fonksiyonlar ---


def get_image_base64(image_name: str) -> str:
    """
    Belirtilen resim dosyasını (script ile aynı dizinde olduğu varsayılarak) okur ve Base64 kodlu bir dize olarak döndürür.
    """
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, image_name)

    print(f"Deniyor: Resim dosyasının yolu: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        print(f"Hata: Resim dosyası bulunamadı: {image_path}")
        return ""
    except Exception as e:
        print(f"Resim okunurken hata oluştu: {e}")
        return ""


def create_emotion_charts_html(emotion_data: dict) -> str:
    """
    Duygu verilerini modern ve şık bir SVG çubuk grafik olarak oluşturur.

    Args:
        emotion_data: Duygu adlarını ve yüzde değerlerini içeren bir sözlük.

    Returns:
        SVG çubuk grafik içeren bir HTML string'i veya veri yoksa bir mesaj.
    """
    labels_map = {
        "duygu_mutlu_%": "Mutlu",
        "duygu_kizgin_%": "Kızgın",
        "duygu_igrenme_%": "İğrenme",
        "duygu_korku_%": "Korku",
        "duygu_uzgun_%": "Üzgün",
        "duygu_saskin_%": "Şaşkın",
        "duygu_dogal_%": "Doğal",
    }

    colors = {
        "Mutlu": "#d4eac8",
        "Kızgın": "#e5b9b5",
        "İğrenme": "#d3cdd7",
        "Korku": "#a9b4c2",
        "Üzgün": "#b7d0e2",
        "Şaşkın": "#fdeac9",
        "Doğal": "#d8d8d8",
    }

    emotion_values = []
    emotion_keys_ordered = [
        "duygu_mutlu_%",
        "duygu_kizgin_%",
        "duygu_igrenme_%",
        "duygu_korku_%",
        "duygu_uzgun_%",
        "duygu_saskin_%",
        "duygu_dogal_%",
    ]

    for key in emotion_keys_ordered:
        if key in emotion_data:
            emotion_name = labels_map.get(key, "Bilinmeyen")
            value = emotion_data.get(key, 0)
            emotion_values.append({"name": emotion_name, "value": value})

    if not emotion_values:
        return "<p>Görselleştirilecek duygu verisi bulunamadı.</p>"

    # Dinamik SVG yüksekliği hesaplama
    base_height = 250  # %100 değerine karşılık gelen yükseklik
    max_value = max(e["value"] for e in emotion_values)
    if max_value < 5:
        max_value = 5  # Çok küçük değerleri engellemek için minimum sınır
    svg_height = int((max_value / 100) * base_height) + 80  # + padding

    svg_width = 600
    padding = 40
    bar_spacing = 15
    label_offset = 5

    num_bars = len(emotion_values)
    bar_width = (svg_width - 2 * padding - (num_bars - 1) * bar_spacing) / num_bars
    if bar_width <= 0:
        bar_width = 20

    svg_elements = []

    # Başlık ekleme
    svg_elements.append(
        f'<text x="{svg_width / 2}" y="25" font-family="IBMPlexSans" font-size="12" text-anchor="middle" fill="#333" font-weight="400">Aday Duygu Analizi</text>'
    )

    # X ekseni çizgisi
    svg_elements.append(
        f'<line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="#ccc" stroke-width="1"/>'
    )

    # Y ekseni etiketleri (0%, 25%, 50%, 75%, 100%)
    for i in range(5):
        percent = i * 25
        y_val = (
            svg_height - padding - ((percent / max_value) * (svg_height - 2 * padding))
        )
        svg_elements.append(
            f'<text x="{padding - 10}" y="{y_val + 5}" font-family="IBMPlexSans" font-size="10" text-anchor="end" fill="#555">{percent}%</text>'
        )
        svg_elements.append(
            f'<line x1="{padding}" y1="{y_val}" x2="{padding + 5}" y2="{y_val}" stroke="#ccc" stroke-width="0.5"/>'
        )

    for i, emotion in enumerate(emotion_values):
        x = padding + i * (bar_width + bar_spacing)
        bar_height = (emotion["value"] / max_value) * (svg_height - 2 * padding)
        y = svg_height - padding - bar_height
        fill_color = colors.get(emotion["name"], "#cccccc")

        svg_elements.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{fill_color}" rx="3" ry="3"/>'
        )

        text_y = y - label_offset
        if text_y < 15:
            text_y = y + 15
            text_fill = "#333"
        else:
            text_fill = "#333"

        svg_elements.append(
            f'<text x="{x + bar_width / 2}" y="{text_y}" font-family="IBMPlexSans" font-size="12" text-anchor="middle" fill="{text_fill}" font-weight="bold">{emotion["value"]:.1f}%</text>'
        )

        svg_elements.append(
            f'<text x="{x + bar_width / 2}" y="{svg_height - padding + 20}" font-family="IBMPlexSans" font-size="11" text-anchor="middle" fill="#555">{emotion["name"]}</text>'
        )

    svg_content = f"""
    <div style="text-align: center; margin: 20px auto; opacity: 0.6;">
        <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" style="background-color: #fcfcfc; border: 1px solid #eee; border-radius: 8px;">
            {''.join(svg_elements)}
        </svg>
    </div>
    """
    return svg_content



def create_emotion_charts_html_2(emotion_data: dict) -> str:
    """
    Çubuk grafiği artık mutlak yüzdeler yerine
    (aday değeri – ortalama) farkıyla çizer.
    Negatif farklar için grafikte aşağı doğru barlar oluşur.
    """
    labels_map = {
        "duygu_mutlu_%": "Mutlu",
        "duygu_kizgin_%": "Kızgın",
        "duygu_igrenme_%": "İğrenme",
        "duygu_korku_%": "Korku",
        "duygu_uzgun_%": "Üzgün",
        "duygu_saskin_%": "Şaşkın",
        "duygu_dogal_%": "Doğal",
    }

    colors = {
        "Mutlu": "#d4eac8",
        "Kızgın": "#e5b9b5",
        "İğrenme": "#d3cdd7",
        "Korku": "#a9b4c2",
        "Üzgün": "#b7d0e2",
        "Şaşkın": "#fdeac9",
        "Doğal": "#d8d8d8",
    }

    # Orijinal sıralama
    emotion_keys = [
        "duygu_mutlu_%",
        "duygu_kizgin_%",
        "duygu_igrenme_%",
        "duygu_korku_%",
        "duygu_uzgun_%",
        "duygu_saskin_%",
        "duygu_dogal_%",
    ]
    avg_keys = [
        "avg_duygu_mutlu_%",
        "avg_duygu_kizgin_%",
        "avg_duygu_igrenme_%",
        "avg_duygu_korku_%",
        "avg_duygu_uzgun_%",
        "avg_duygu_saskin_%",
        "avg_duygu_dogal_%",
    ]

    # Farkları hesapla
    diffs = []
    for key, avg_key in zip(emotion_keys, avg_keys):
        name = labels_map[key]
        val = emotion_data.get(key, 0)
        avg = emotion_data.get(avg_key, 0)
        diff = round(val - avg, 2)
        diffs.append({"name": name, "value": diff})

    if not diffs:
        return "<p>Görselleştirilecek duygu verisi bulunamadı.</p>"

    # Ölçek: en büyük mutlak fark
    max_abs = max(abs(d["value"]) for d in diffs)
    if max_abs < 5:
        max_abs = 5

    # Grafik ölçüleri (üst + alt için simetrik olacak şekilde)
    base_height = 250
    padding = 40
    panel = (max_abs / 100) * base_height
    svg_height = int(panel * 2 + padding * 2)
    svg_width = 600
    bar_spacing = 15
    num_bars = len(diffs)
    bar_width = (svg_width - 2 * padding - (num_bars - 1) * bar_spacing) / num_bars

    # Sıfır hattı (baseline) orta noktada
    baseline_y = padding + panel

    svg_elems = []

    # Başlık ekleme
    svg_elems.append(
        f'<text x="{svg_width / 2}" y="25" font-family="IBMPlexSans" font-size="12" text-anchor="middle" fill="#333" font-weight="400">Aday Duygularının Ortalamadan Farkı</text>'
    )

    # Y=0 hattı
    svg_elems.append(
        f'<line x1="{padding}" y1="{baseline_y}" x2="{svg_width-padding}" '
        f'y2="{baseline_y}" stroke="#ccc" stroke-width="1"/>'
    )

    # Y ekseni etiketleri (negatiften pozitife)
    for perc in [-max_abs, 0, max_abs]:
        # yüzde etiketlerimizi -X%, -X/2%, 0%, +X/2%, +X% olarak koyabiliriz
        # tam -100…100 arasında etiketlemek yerine göreceli ölçek
        pos = baseline_y - (perc / max_abs) * panel
        label = f"{perc:.0f}%"
        svg_elems.append(
            f'<text x="{padding-10}" y="{pos+4}" font-family="IBMPlexSans" '
            f'font-size="10" text-anchor="end" fill="#555">{label}</text>'
        )
        svg_elems.append(
            f'<line x1="{padding}" y1="{pos}" x2="{padding+5}" y2="{pos}" '
            f'stroke="#ccc" stroke-width="0.5"/>'
        )

    # Barlar
    for i, item in enumerate(diffs):
        x = padding + i * (bar_width + bar_spacing)
        val = item["value"]
        height = abs(val) / max_abs * panel
        if val >= 0:
            y = baseline_y - height
        else:
            y = baseline_y
        color = colors.get(item["name"], "#ccc")
        svg_elems.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{height}" '
            f'fill="{color}" rx="3" ry="3"/>'
        )
        # Değer etiketleri
        txt_y = y - 5 if val >= 0 else y + height + 15
        svg_elems.append(
            f'<text x="{x+bar_width/2}" y="{txt_y}" font-family="IBMPlexSans" '
            f'font-size="12" text-anchor="middle" fill="#333" font-weight="bold">'
            f"{val:+.1f}%</text>"
        )
        # Duygu adı
        svg_elems.append(
            f'<text x="{x+bar_width/2}" y="{baseline_y + panel + 20}" '
            f'font-family="IBMPlexSans" font-size="11" text-anchor="middle" fill="#555">'
            f'{item["name"]}</text>'
        )

    svg = (
        f'<div style="text-align:center;margin:20px auto;opacity:0.6;">'
        f'<svg width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}" '
        f'style="background-color:#fcfcfc;border:1px solid #eee;border-radius:8px;">'
        + "".join(svg_elems)
        + "</svg></div>"
    )
    return svg


def format_qa_section(qa_list: list) -> str:
    """
    Soru-cevap listesini okunabilir bir HTML formatına dönüştürür.
    """
    html = ""
    for item in qa_list:
        html += f"""
        <div class="qa-item" style="margin-bottom: 15px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 8px;">
            <p style="font-weight: bold; color: #34495e;">Soru: {item['soru']}</p>
            <p style="color: #555; margin-top: 5px;">Cevap: {item['cevap']}</p>
        </div>
        """
    return html


def generate_llm_prompt(row_data: dict, formatted_qa_html: str) -> str:
    """
    Verilen toplu veri satırına ve yeni, daha temiz bir HTML şablonuna dayanarak Gemini LLM için prompt oluşturur.
    Filigran resmi LLM'e gönderilmez, sonradan eklenecektir.
    """

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @font-face {{
            font-family: "IBMPlexSans";
            src: url("fonts/IBMPlexSans-Regular.ttf");
            font-weight: normal;
            font-style: normal;
        }}
        @font-face {{
            font-family: "IBMPlexSans";
            src: url("fonts/IBMPlexSans-Medium.ttf");
            font-weight: 500;
            font-style: normal;
        }}
        @font-face {{
            font-family: "IBMPlexSans";
            src: url("fonts/IBMPlexSans-Bold.ttf");
            font-weight: bold;
            font-style: normal;
        }}
        body {{
            font-family: "IBMPlexSans", sans-serif;
            line-height: 1.7;
            margin: 25px;
            color: #333;
            background-color: #ffffff;
            font-size: 10pt;
            position: relative;
            margin-bottom: 40px;
            width: 100vw;
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 10px; 
            font-size: 24px; 
            font-weight: bold;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 35px; 
            border-bottom: 1px solid #bdc3c7; 
            padding-bottom: 8px; 
            font-size: 20px;
            font-weight: 500;
        }}
        h3 {{ 
            color: #7f8c8d; 
            font-size: 16px; 
            margin-bottom: 15px; 
            font-weight: 500;
        }}
        .section {{ margin-bottom: 30px; }}
        #pie-chart-placeholder {{ width: 100%; height: auto; margin: 20px auto; text-align: center; }}

        /* Filigran Resim Konteyneri */
        .watermark-image-container {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: -1;
            pointer-events: none;
            opacity: 0.05;
            width: 70%;
            max-width: 600px;
            height: auto;
            text-align: center;
        }}
        .watermark-image-container img {{
            width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}

        @page {{
            margin: 70px 12.5px 70px 12.5px;
            @top-left {{
                content: element(header_logo);
                vertical-align: top;
            }}
            @top-right {{
                content: element(header_uygunluk);
                vertical-align: top;
            }}
            @bottom-center {{
                content: element(footer_content);
                vertical-align: bottom;
                padding-bottom: 10px;
            }}
        }}    


        /* Alt bilgi stili */
        .page-footer {{
            display: block;
            position: running(footer_content);
            width: 100%;
            background-color: #ffffff;
            padding: 10px 10px;
            text-align: center;
            font-size: 8px;
            color: #555;
            box-sizing: border-box;
        }}
        .footer-divider {{
            border-top: 0.5px solid #ccc;
            margin: 0 auto 5px auto;
            width: 90%;
        }}
        .footer-company-name {{
            font-weight: bold;
            margin-bottom: 2px;
        }}
        .footer-contact-info {{
            font-size: 7px;
            line-height: 1.2;
            white-space: nowrap;
            display: flex;
            justify-content: center;
            gap: 10px;
        }}

        /* LOGO HEADER - Sol üst */
        .page-header-logo {{
            margin-top: 15px;
            margin-left: 15px;
            position: running(header_logo);
            text-align: left;
        }}
        .page-header-logo img {{
            width: 40px;
            height: auto;
            display: inline-block;
        }}

        /* POZISYONA UYGUNLUK - Sağ üst köşe */
        .page-header-uygunluk {{
            margin-top: 15px;
            margin-right: 15px;
            position: running(header_uygunluk);
            text-align: right;
            font-size: 12px;
            color: #2c3e50;
            font-weight: bold;
            line-height: 1.2;
            min-width: 180px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #27ae60;
            border-radius: 8px;
            padding: 8px 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .page-header-uygunluk .percentage {{
            font-size: 16px;
            color: #27ae60;
            font-weight: bold;
            display: inline-block;
            margin-left: 5px;
        }}

        .page-header-uygunluk .info-item {{
            margin-bottom: 2px;
        }}
        .page-header-uygunluk .info-link {{
            text-decoration: none;
            color: #223;
            font-weight: bold;
        }}
        .page-header-uygunluk .icon {{
            font-size: 13px;
            margin-right: 4px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <!-- Logo header elementi -->
    <div class="page-header-logo" id="header_logo">
        <img src="{{{{logo_src}}}}" alt="Logo" />
    </div>
    
    <!-- Sağ üst köşe: pozisyona uygunluk bilgisi -->
    <div class="page-header-uygunluk" id="header_uygunluk">
        Pozisyona Uygunluk: <span class="percentage">%{row_data['llm_skoru']}</span>
    </div>

    <!-- Alt bilgi elementi -->
    <div class="page-footer">
        <div class="footer-divider"></div>
        <div class="footer-company-name">DeepWork Bilişim Teknolojileri A.Ş.</div>
        <div class="footer-contact-info">
            <span>info@hrai.com.tr</span>
            <span>-</span>
            <span>İstanbul Medeniyet Üniversitesi Kuzey Kampüsü Medeniyet Teknopark Kuluçka Merkezi Üsküdar/İstanbul</span>
            <span>-</span>
            <span>+90 553 808 32 77</span>
        </div>
    </div>
    
    <!-- Filigran Resim Konteyneri -->
    <div class="watermark-image-container" id="watermark-placeholder">
        <!-- Resim buraya dinamik olarak eklenecek -->
    </div>
    
    <h1>{row_data['kisi_adi']} - Mülakat Değerlendirme Raporu</h1>
    
    <div class="section">
        <h2>1) Genel Bakış</h2>
        <p>{{{{genel_bakis_icerik}}}}</p>
    </div>
    
    <div class="section">
        <h2>2) Analiz</h2>
        <h3>Duygu Analizi:</h3>
        <div id="bar-chart-placeholder"></div> <p>{{{{duygu_analizi_yorumu}}}}</p>

        <h3>Dikkat Analizi</h3>
        <p>{{{{dikkat_analizi_yorumu}}}}</p>
    </div>
    
    <div class="section">
        <h2>3) Genel Değerlendirme</h2>
        <p>{{{{genel_degerlendirme_icerik}}}}</p>
    </div>

    <div class="section">
        <h2>4) Sorular ve Cevaplar</h2>
        {formatted_qa_html}
    </div>
    
    <div class="section">
        <h2>5) Sonuçlar ve Öneriler</h2>
        <p>{{{{sonuclar_oneriler_icerik}}}}</p>
    </div>

    {{{{uygunluk_degerlendirmesi_bolumu}}}}
</body>
</html>
"""



    if row_data["tip"] == 0:
        # DEĞİŞTİRİLEN KISIM: Aday Uygunluk Bölümü Eklendi
        prompt_instructions = f"""
Lütfen aşağıdaki HTML şablonunu verilen mülakat verilerine göre doldurarak eksiksiz bir HTML raporu oluştur.
Veriler:
- Aday Adı: {row_data['kisi_adi']}
- Mülakat Adı: {row_data['mulakat_adi']}
- LLM Skoru: {row_data['llm_skoru']}, Ortalama LLM Skoru: {row_data['avg_llm_skoru']}
- Duygu Analizi (%): Mutlu {row_data['duygu_mutlu_%']}, Kızgın {row_data['duygu_kizgin_%']}, İğrenme {row_data['duygu_igrenme_%']}, Korku {row_data['duygu_korku_%']}, Üzgün {row_data['duygu_uzgun_%']}, Şaşkın {row_data['duygu_saskin_%']}, Doğal {row_data['duygu_dogal_%']}
- Dikkat Analizi: Ekran Dışı Süre {row_data['ekran_disi_sure_sn']} sn, Ekran Dışı Bakış Sayısı {row_data['ekran_disi_sayisi']}, Ortalama Ekran Dışı Süre {row_data['avg_ekran_disi_sure_sn']} sn, Ortalama Ekran Dışı Bakış Sayısı {row_data['avg_ekran_disi_sayisi']}

Doldurulacak Alanlar İçin Talimatlar:
1.  `{{{{genel_bakis_icerik}}}}`: Adayın genel performansını, iletişim becerilerini ve mülakatın genel seyrini özetleyen, en az iki paragraftan oluşan detaylı bir giriş yaz.
2.  `{{{{duygu_analizi_yorumu}}}}`: Yukarıda verilen sayısal duygu analizi verilerini yorumla. Hangi duyguların baskın olduğunu ve bunun mülakat bağlamında ne anlama gelebileceğini analiz et. Bu yorum en az iki detaylı paragraf olmalıdır. Giriş cümlesi tam olarak şu olmalı: "Görüntü ve ses analiz edilerek adayın duygu analizi yapılmıştır."
3.  `{{{{dikkat_analizi_yorumu}}}}`: Ekran dışı süre ve bakış sayısı verilerini yorumla. Bu verilerin adayın dikkat seviyesi veya odaklanması hakkında ne gibi ipuçları verdiğini açıkla. Bu yorum en az bir detaylı paragraf olmalıdır.
4.  `{{{{genel_degerlendirme_icerik}}}}`: Adayın verdiği cevapları, genel tavrını ve analiz sonuçlarını birleştirerek kapsamlı bir değerlendirme yap. Adayın güçlü ve gelişime açık yönlerini belirt. Bu bölüm en az üç paragraf olmalıdır.
5.  `{{{{sonuclar_oneriler_icerik}}}}`: Bu bölümü **sadece İnsan Kaynakları profesyonellerine yönelik** olarak yaz. Adayın pozisyona uygunluğu hakkında net bir sonuca var. İşe alım kararı için somut önerilerde bulun. Adaya yönelik bir dil kullanma. Bu bölüm en az iki paragraf olmalıdır.
6.  **YENİ TALİMAT**: `{{{{uygunluk_degerlendirmesi_bolumu}}}}`: Adayın pozisyona uygunluk yüzdesini (0-100 arası bir tam sayı) ve bu yüzdeyi destekleyen kısa bir açıklamayı HTML formatında oluştur. Yüzdeyi `{row_data['llm_skoru']}` değerini dikkate alarak belirle. Örnek format:
    ```html
    <div class="section">
        <h2>6) Pozisyona Uygunluk Değerlendirmesi</h2>
        <p style="font-size: 18px; font-weight: bold; color: #27ae60;">Pozisyona Uygunluk: %85</p>
        <p>Adayın genel mülakat performansı, teknik bilgi ve iletişim becerileri, pozisyonun gerektirdiği yetkinliklerle yüksek düzeyde örtüşmektedir. Duygu analizi ve dikkat seviyesi de olumlu bir tablo çizmektedir.</p>
    </div>
    ```
    Yüzdeyi ve açıklamayı doldururken, verilen LLM Skoru'nu doğrudan uygunluk yüzdesi olarak kullanabilir veya bu skora dayanarak mantıklı bir uygunluk yüzdesi türetebilirsin. Açıklama 1-2 paragraf uzunluğunda olmalıdır.

Önemli Kurallar:
- Üretilen tüm metin **sadece Türkçe** olmalıdır.
- Raporun tonu profesyonel, resmi ve veri odaklı olmalıdır.
- Kullanıcıya yönelik hiçbir not, açıklama veya meta-yorum ekleme.
- Sadece ve sadece aşağıdaki HTML şablonunu doldurarak yanıt ver. Başka hiçbir metin ekleme.

İşte doldurman gereken şablon:
{html_template}
"""

    elif row_data["tip"] == 1:
        # Müşteri raporu için uygunluk bölümünü boş bırakın
        prompt_instructions = f"""
Lütfen aşağıdaki HTML şablonunu verilen mülakat verilerine göre doldurarak eksiksiz bir HTML raporu oluştur.
Veriler:
- Müşteri Adı: {row_data['kisi_adi']}
- Görüşme Adı: {row_data['mulakat_adi']}
- Duygu Analizi (%): Mutlu {row_data['duygu_mutlu_%']}, Kızgın {row_data['duygu_kizgin_%']}, İğrenme {row_data['duygu_igrenme_%']}, Korku {row_data['duygu_korku_%']}, Üzgün {row_data['duygu_uzgun_%']}, Şaşkın {row_data['duygu_saskin_%']}, Doğal {row_data['duygu_dogal_%']}
- Dikkat Analizi: Ekran Dışı Süre {row_data['ekran_disi_sure_sn']} sn, Ekran Dışı Bakış Sayısı {row_data['ekran_disi_sayisi']}, Ortalama Ekran Dışı Süre {row_data['avg_ekran_disi_sure_sn']} sn, Ortalama Ekran Dışı Bakış Sayısı {row_data['avg_ekran_disi_sayisi']}

Doldurulacak Alanlar İçin Talimatlar:
1.  `{{{{genel_bakis_icerik}}}}`: Müşterinin genel performansını, iletişim becerilerini ve görüşmenin genel seyrini özetleyen, en az iki paragraftan oluşan detaylı bir giriş yaz.
2.  `{{{{duygu_analizi_yorumu}}}}`: Yukarıda verilen sayısal duygu analizi verilerini yorumla. Hangi duyguların baskın olduğunu ve bunun görüşme bağlamında ne anlama gelebileceğini analiz et. Bu yorum en az iki detaylı paragraf olmalıdır. Giriş cümlesi tam olarak şu olmalı: "Görüntü ve ses analiz edilerek kişinin duygu analizi yapılmıştır."
3.  `{{{{dikkat_analizi_yorumu}}}}`: Ekran dışı süre ve bakış sayısı verilerini yorumla. Bu verilerin müşterinin dikkat seviyesi veya odaklanması hakkında ne gibi ipuçları verdiğini açıkla. Bu yorum en az bir detaylı paragraf olmalıdır.
4.  `{{{{genel_degerlendirme_icerik}}}}`: Müşterinin verdiği cevapları, genel tavrını ve analiz sonuçlarını birleştirerek kapsamlı bir değerlendirme yap. Müşterinin güçlü ve gelişime açık yönlerini belirt. Bu bölüm en az üç paragraf olmalıdır.
5.  `{{{{sonuclar_oneriler_icerik}}}}`: Bu bölümü müşteri hakkında genel bir değerlendirme olarak yaz. 1 paragraf kadar olmalı

Önemli Kurallar:
- Üretilen tüm metin **sadece Türkçe** olmalıdır.
- Raporun tonu profesyonel, resmi ve veri odaklı olmalıdır.
- Kullanıcıya yönelik hiçbir not, açıklama veya meta-yorum ekleme.
- Sadece ve sadece aşağıdaki HTML şablonunu doldurarak yanıt ver. Başka hiçbir metin ekleme.

İşte doldurman gereken şablon:
{html_template}
"""

    return prompt_instructions


def create_pdf_from_html(html_content: str) -> io.BytesIO:
    """
    Bir HTML dizesinden WeasyPrint kullanarak bir PDF dosyası oluşturur.
    Fontlar gibi yerel dosyalara erişim için bir base_url kullanır.
    """
    try:
        pdf_buffer = io.BytesIO()
        html = HTML(string=html_content, base_url=".")
        html.write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        print(f"WeasyPrint PDF oluşturulurken hata: {e}")
        raise ValueError(f"PDF oluşturulurken WeasyPrint hatası oluştu: {e}")


# --- FastAPI Endpoint'i ---


@app.post("/generate-report", summary="PDF Mülakat Raporu Oluştur")
async def generate_report(
    file: UploadFile = File(..., description="Mülakat verilerini içeren CSV dosyası.")
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Hatalı dosya formatı. Lütfen bir .csv dosyası yükleyin.",
        )

    try:
        file_content = await file.read()
        df = pd.read_csv(io.BytesIO(file_content))

        required_columns = [
            "kisi_adi",
            "mulakat_adi",
            "llm_skoru",
            "duygu_mutlu_%",
            "duygu_kizgin_%",
            "duygu_igrenme_%",
            "duygu_korku_%",
            "duygu_uzgun_%",
            "duygu_saskin_%",
            "duygu_dogal_%",
            "ekran_disi_sure_sn",
            "ekran_disi_sayisi",
            "soru",
            "cevap",
            "tip",
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise HTTPException(
                status_code=400,
                detail=f"CSV dosyasında eksik sütunlar var: {', '.join(missing_cols)}",
            )

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV dosyası veri içermiyor.")

        row = df.iloc[0]

        current_row_data = {
            "kisi_adi": row["kisi_adi"],
            "mulakat_adi": row["mulakat_adi"],
            "llm_skoru": round(row["llm_skoru"], 2),
            "duygu_mutlu_%": round(row["duygu_mutlu_%"], 2),
            "avg_duygu_mutlu_%": round(row["avg_duygu_mutlu_%"], 2),
            "duygu_kizgin_%": round(row["duygu_kizgin_%"], 2),
            "avg_duygu_kizgin_%": round(row["avg_duygu_kizgin_%"], 2),
            "duygu_igrenme_%": round(row["duygu_igrenme_%"], 2),
            "avg_duygu_igrenme_%": round(row["avg_duygu_igrenme_%"], 2),
            "duygu_korku_%": round(row["duygu_korku_%"], 2),
            "avg_duygu_korku_%": round(row["avg_duygu_korku_%"], 2),
            "duygu_uzgun_%": round(row["duygu_uzgun_%"], 2),
            "avg_duygu_uzgun_%": round(row["avg_duygu_uzgun_%"], 2),
            "duygu_saskin_%": round(row["duygu_saskin_%"], 2),
            "avg_duygu_saskin_%": round(row["avg_duygu_saskin_%"], 2),
            "duygu_dogal_%": round(row["duygu_dogal_%"], 2),
            "avg_duygu_dogal_%": round(row["avg_duygu_dogal_%"], 2),
            "ekran_disi_sure_sn": round(row["ekran_disi_sure_sn"], 2),
            "avg_ekran_disi_sure_sn": round(row["avg_ekran_disi_sure_sn"], 2),
            "ekran_disi_sayisi": int(row["ekran_disi_sayisi"]),
            "avg_ekran_disi_sayisi": int(row["avg_ekran_disi_sayisi"]),
            "soru_cevap": [{"soru": row["soru"], "cevap": row["cevap"]}],
            "tip": int(row["tip"]),
            "avg_llm_skoru": round(row["avg_llm_skoru"], 2),
        }

        print(f"İşlenen satır tipi: {current_row_data['tip']}")

        formatted_qa_html = format_qa_section(current_row_data["soru_cevap"])

        prompt = generate_llm_prompt(current_row_data, formatted_qa_html)

        response = gemini_model.generate_content(
            prompt, generation_config=genai.types.GenerationConfig(temperature=0.7)
        )

        raw_html_content = (
            response.text.strip().removeprefix("```html").removesuffix("```")
        )

        soup = BeautifulSoup(raw_html_content, "html.parser")

        # Duygu analizi grafiği yer tutucusunu güncelle:
        #   1) Mutlak değerler grafiği
        #   2) Ortalama fark grafiği
        bar_chart_placeholder = soup.find(id="bar-chart-placeholder")
        if bar_chart_placeholder:
            # 1) Mutlak duygu yüzdeleri
            abs_chart_html = create_emotion_charts_html(current_row_data)
            # 2) Aday–ortalama farkı
            diff_chart_html = create_emotion_charts_html_2(current_row_data)

            bar_chart_placeholder.clear()
            bar_chart_placeholder.append(
                BeautifulSoup(abs_chart_html + diff_chart_html, "html.parser")
            )

        logo_base64 = get_image_base64("logo.png")
        if logo_base64:
            logo_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

            # 1) Header logosunu ayarla
            header_img = soup.select_one("#header_logo img")
            if header_img and logo_src:
                header_img["src"] = logo_src

            # 2) Filigran logosunu ayarla
            watermark_placeholder = soup.find(id="watermark-placeholder")
            if watermark_placeholder:
                img_tag = soup.new_tag(
                    "img", src=logo_src, alt="Deepwork Logo Filigranı"
                )
                watermark_placeholder.append(img_tag)
        else:
            print("Uyarı: logo.png bulunamadı veya okunamadı. Filigran eklenemedi.")

        # YENİ EKLENEN: tip 1 ise uygunluk bölümünü HTML'den tamamen kaldır
        if current_row_data["tip"] == 1:
            uygunluk_placeholder = soup.find(text="{{uygunluk_degerlendirmesi_bolumu}}")
            if uygunluk_placeholder:
                uygunluk_placeholder.extract()  # Placeholdere bağlı metni kaldır

        final_html = soup.prettify()

        html_debug_filename = f"{current_row_data['kisi_adi']}_{current_row_data['mulakat_adi']}_Rapor_Debug.html"
        try:
            with open(html_debug_filename, "w", encoding="utf-8") as f:
                f.write(final_html)
            print(f"HTML içeriği '{html_debug_filename}' dosyasına kaydedildi.")
        except IOError as io_err:
            print(f"HTML içeriği kaydedilirken hata oluştu: {io_err}")

        pdf_bytes = create_pdf_from_html(final_html)

        filename = f"{current_row_data['kisi_adi']}_{current_row_data['mulakat_adi']}_Rapor.pdf"
        encoded_filename = urllib.parse.quote(filename)

        return StreamingResponse(
            pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            },
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Yüklenen CSV dosyası boş.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Rapor oluşturulurken sunucuda bir hata oluştu: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)