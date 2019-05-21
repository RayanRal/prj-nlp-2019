Для курсового проекту мені може знадобитися як створити додаткові дані (як власноруч так і за допомогою краудсорсингу) у вигляді анотацій сутностей так і скористатися вже існуючими даними (dbpedia, wikipedia і т.д). Для першого знадобиться створити інтерфейс для анотування сутностей у браузері, для другого - написання кравлеру.

Прізвиська головних героїв книжок/серіалу "Гра Престолів", зібрані з Вікіпедії як доказ концепції.

```
{
  'Ned Stark': ['The Quiet Wolf'],
  'Robert Baratheon':['The Usurper', 'Demon of the Trident', 'The Whoremonger King', 'The Stag King'],
  'Jaime Lannister': ['The Kingslayer', 'The Young Lion'],
  'Catelyn Stark':['Cat', 'Lady Cat', 'The Silent Sister', 'Mother Merciless', 'The Hangwoman', 'Lady Stonehart'],
  'Cersei Lannister': [],
  'Daenerys Targaryen':['Daenerys Stormborn', 'Dany', 'Khaleesi', 'Mhysa', 'The Silver Queen', 'Silver Lady', 'Dragonmother', 'The Dragon Queen', 'The Queen Across the Water', 'The Princess that was Promised'],
  'Jorah Mormont': ['Jorah the Andal'],
  'Viserys Targaryen': ['The Beggar King', 'The Sorefoot King', 'The Cart King'],
  'Jon Snow (character)':['Lord Snow', 'The Bastard of Winterfell', 'The Snow of Winterfell', 'The crow-come-over', 'Lord Crow', 'The Black Bastard of the Wall', 'King Crow', 'The White Wolf', 'The Prince That Was Promised', 'Aegon Targaryen (birth name)'],
  'Sansa Stark':['Little Bird', 'Little Dove', 'Alayne Stone', 'Jonquil', 'Elaine'],
  'Arya Stark':['Arya Horseface', 'Arya Underfoot', 'arry', 'Lumpyhead', 'Lumpyface', 'Stickboy', 'Rabbitkiller', 'Weasel', 'Nymeria', 'Nan', 'Squab', 'Squirrel', 'Blood Child', 'Wolf Girl', 'Salty', 'No One', 'Night Wolf', 'Blind Beth', 'The Blind Girl', 'The Gorgeous girl', 'Mercedene', 'Mercy', 'Cat of the Canals', 'Lanna of the Canals'],
  'Robb Stark':['The Young Wolf', 'The King Who Lost the North', 'Robb the Lord', 'The Wolfling', 'The Pup', 'The Boy Wolf', 'The Wolf Pup'],
  'Theon Greyjoy':['Prince of Fools', 'Theon Turncloak', 'The Squid Prince', 'Reek', 'Theon Kinslayer', 'The Prince of Stink'],
  'Bran Stark':['Bran the Broken', 'The Winged Wolf', 'Little Lord', 'Three-Eyed Raven'],
  'Joffrey Baratheon': ['Joffrey the Illborn', 'The Young Usurper'],
  'Sandor Clegane': ['The Hound', 'Dog'],
  'Tyrion Lannister': ['The Imp', 'The Halfman', 'Yollo', 'Hugor Hill'],
  'Khal Drogo': ['Great Rider', 'Great Khal'],
  'Petyr Baelish': ['Littlefinger'],
  'Davos Seaworth':['The Onion Knight', 'Davos Shorthand', 'Davos the Smuggler', 'Davos of Flea Bottom', 'Knight of the Onion Ship', 'Ser Onions', 'Onion Lord'],
  'Samwell Tarly':['Sam', 'Ser Piggy', 'Lady Piggy', 'Lord of Ham', 'Sam the Slayer', 'Black Sam'],
  'Stannis Baratheon':['The King in the Narrow Sea', 'The King of the Painted Table', 'The King of Dragonstone', 'The King at the Wall'],
  Melisandre: ['The Red Priestess', 'The Red Woman'],
  'List of A Song of Ice and Fire characters': [],
  'Bronn (character)': ['Ser Bronn of the Blackwater', 'The Cutthroat'],
  Varys:['The Spider', 'The Eunuch', 'Lord Varys', 'Rugen', 'Varys of Lys'],
  'List of Game of Thrones characters': [],
  'Margaery Tyrell': ['The Little Queen', 'The Little Rose', 'Maid Margaery'],
  'Tywin Lannister':['The Lion of Lannister', 'The Old Lion', 'The Great Lion of the Rock'],
  Ygritte: [],
  Gendry: ['The Bull', 'Ser Gendry of the hollow hill', 'Clovis'],
  'Tormund Giantsbane':['Tormund Giantsbane', 'Tormund Giantsbabe', 'Tormund Giantstink', 'Mead-King of Ruddy Hall', 'Tormund Thunderfist', 'Tormund Horn-Blower', 'Tormund Tall-Talker', 'Breaker of Ice', 'Husband to Bears', 'Speaker to Gods', 'Father of Hosts'],
  'Brienne of Tarth':['Lady Brienne', 'The Maid of Tarth', 'Brienne the Beauty', 'Brienne the Blue'],
  'Ramsay Bolton':['The Bastard of Bolton', 'Ramsay Snow', 'The Bastard of the Dreadfort', 'Red Helm', 'Lord Snow'],
  'Gilly (character)': ['The Rabbit Keeper'],
  'Daario Naharis': [],
  Missandei: [],
  'Ellaria Sand': [ "The Serpent's Whore" ],
  'Tommen Baratheon': ['The Boy King'],
  'Roose Bolton': ['The Leech Lord', 'Lord Leech'],
  'High Sparrow': []
}
```
