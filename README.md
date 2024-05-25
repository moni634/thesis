# Program ImageComparator
Aplikácia ImageComparatorApp je grafická aplikácia vyvinutá v Pythone s použitím knižnice tkinter. Jej hlavným účelom je porovnávanie dvoch obrázkov na základe rôznych metód, ako sú rozdielové obrázky, detekcia bodov a hrán, výpočet SSIM a analýza histogramov.

## Trieda ImageComparatorApp
### Inicializácia
Metóda **\__init__** inicializuje hlavnú aplikáciu a jej komponenty.

### Načítanie obrázkov
Metódy load_image1 a load_image2 umožňujú používateľovi vybrať a načítať dva obrázky.

### Resizing obrázkov
Metódy resize_image a resize_image_for_display upravujú veľkosť obrázkov pre zobrazovanie.

### Porovnávanie obrázkov
Metóda compare_images porovnáva štatistické údaje dvoch obrázkov a zobrazuje ich v textových poliach.

### Výpočet štatistík
Metóda calculate_stats vypočíta základné štatistické údaje obrázka ako priemer, štandardná odchýlka, medián, minimum a maximum v RGB a grayscale formáte.

### Zobrazenie rozdielového obrázka
Metóda show_difference_image zobrazuje rozdielový obrázok medzi dvoma obrázkami.

### Zobrazenie histogramov
Metódy show_rgb_histograms a show_combined_histogram zobrazujú histogramy intenzít pixelov pre RGB a grayscale formáty.

### Detekcia bodov a hrán
Okno pre detekciu
Metóda harris_corner_window otvára nové okno pre zadanie parametrov detekcie Harrisových rohov.

**Meniteľné parametre:**
* block_size: Určuje veľkosť oblasti, nad ktorou sa vypočítavajú vlastnosti pre
detekciu bodov. Ide o susedné okolie pre každý pixel s veľkosťou block_size ×
block_size
* ksize: Veľkosť Sobelovho operátora použitého na výpočet derivácií obrazu.
* k: Parameter Harrisovej rovnice, ktorý ovplyvňuje citlivosť na hrany a rohy.
* threshold: Prahová hodnota, ktorá určuje, ktoré body budú považované za vý-
znamné. Znížením tejto hodnoty bude detekované väčšie množstvo bodov.

**Resetovanie parametrov**
Metóda reset_parameters obnoví pôvodné hodnoty parametrov pre detekciu.

**Spustenie detekcie**
Metóda start_harris_corner_detection spúšťa detekciu Harrisových rohov v novom vlákne a zobrazuje priebeh operácie.

**Detekcia Harrisových rohov**
Metóda harris_corner_detection vykonáva samotnú detekciu rohov a zobrazuje výsledok na canvas.

**Pomocné metódy**
Metódy detect_harris_corners, draw_corners, connect_regions, update_image_on_canvas, a show_image_on_canvas zabezpečujú detekciu rohov, ich zobrazenie a pripojenie jednotlivých bodov.

### Výpočet SSIM
Metóda calculate_ssim vypočíta štrukturálny index podobnosti (SSIM) medzi dvoma obrázkami a zobrazí výsledok.

Použitie
* Spustite aplikáciu.
* Kliknite na tlačidlo "Hľadaj" vedľa položky "Vyber obrázok 1" a vyberte prvý obrázok.
* Kliknite na tlačidlo "Hľadaj" vedľa položky "Vyber obrázok 2" a vyberte druhý obrázok.
* Kliknite na tlačidlo "Porovnaj" pre zobrazenie základných štatistík obrázkov.
* Použite tlačidlá v rámci na vykonanie ďalších operácií, ako je zobrazenie rozdielového obrázka, detekcia rohov, výpočet SSIM, a zobrazenie histogramov.


Použité knižnice:
* tkinter: Na vytvorenie GUI aplikácie.
* PIL (Pillow): Na manipuláciu s obrázkami.
* cv2 (OpenCV): Na spracovanie obrázkov.
* numpy: Na numerické operácie.
* matplotlib: Na zobrazovanie histogramov.
* seaborn: Na zlepšenie vzhľadu grafov.
* sklearn: Na použitie algoritmu DBSCAN.
* scipy: Na optimalizáciu priradenia liniek medzi bodmi.
* skimage: Na výpočet SSIM.


