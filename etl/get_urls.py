import json
import random
import time
import requests
from bs4 import BeautifulSoup

def get_all_boxscore_urls(start_season=2022, end_season=2026):
    teams = [
        'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]
    
    unique_boxscores = set()
    seasons = list(range(start_season, end_season + 1))
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"--- {start_season} - {end_season} Sezonları Fikstür Taraması Başladı ---")
    
    for season in seasons:
        print(f"\n==========================================")
        print(f"  {season} SEZONU TARANIYOR")
        print(f"==========================================")
        
        for index, team in enumerate(teams, 1):
            url = f"https://www.basketball-reference.com/teams/{team}/{season}_games.html"
            print(f"[{season} | {index}/30] {team} fikstürü taranıyor...")
            
            try:
                res = requests.get(url, headers=headers)
                
                # Rate limit koruması (429 olursa durup bekle)
                if res.status_code == 429:
                    print("[RATE LIMIT] 429 yanıtı alındı! 30 saniye bekleniyor...")
                    time.sleep(30)
                    res = requests.get(url, headers=headers)
                
                if res.status_code == 200:
                    soup = BeautifulSoup(res.content, "html.parser")
                    # data-stat='box_score_text' olan td hücrelerindeki linkleri topla
                    box_cells = soup.find_all("td", {"data-stat": "box_score_text"})
                    
                    for cell in box_cells:
                        link = cell.find("a")
                        if link and "href" in link.attrs:
                            full_url = "https://www.basketball-reference.com" + link["href"]
                            unique_boxscores.add(full_url)
                            
            except Exception as e:
                print(f"[HATA] {team} ({season}) çekilirken hata oluştu: {e}")
                
            print(f"-> Şu ana kadar toplanan toplam benzersiz maç: {len(unique_boxscores)}")
            
            # Basketball-Reference isteği engellemesin diye bekleme süresi
            time.sleep(random.uniform(2.5, 3.8))
            
    boxscore_list = sorted(list(unique_boxscores))
    output_filename = "all_boxscore_urls.json"
    
    with open(output_filename, "w") as f:
        json.dump(boxscore_list, f, indent=4)
        
    print(f"\n[İŞLEM TAMAMLANDI] {len(boxscore_list)} benzersiz maç linki '{output_filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    # 2022, 2023, 2024, 2025 ve 2026 sezonlarını tarar
    get_all_boxscore_urls(start_season=2022, end_season=2026)