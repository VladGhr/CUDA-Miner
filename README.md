## Construirea Merkle root ului

Functia construct_merkle_root construieste arborele Merkle printr-o reducere
binara: nivelul 0 contine SHA256 ul fiecarei tranzactii, iar fiecare nivel
superior combina perechi de hash uri pana ramane un singur hash (root ul).

### Kernel ul merkle_level0_kernel

Foloseste un thread per tranzactie. Fiecare thread calculeaza index = blockIdx.x *
blockDim.x + threadIdx.x, iar daca indexul depaseste numarul de tranzactii
n, thread ul se opreste. Altfel aplica  apply_sha256 pe tranzactia de la
offset-ul  index * transaction_size si scrie rezultatul in vectorul de
hash-uri de pe GPU.

In urma acestui kernel obtinem n hash uri, fiecare de lungime
SHA256_HASH_SIZE, stocate direct in memoria gpu ului.

### Kernel merkle_reduce_kernel

Acest kernel realizeaza un singur nivel de reducere si este lansat repetat din
host pana cand mai ramane un singur hash. Un thread proceseaza o pereche de
hash uri, deci sunt necesare ceil(n / 2) thread uri.

Fiecare thread:

   Calculeaza indicii i = tid * 2 si j = i + 1. Daca j depaseste n
   (numar impar de hash-uri pe nivelul curent), j devine egal cu i, ceea
   ce duplica ultimul hash, apoi concateneaza cele doua hash uri intr-un buffer local combined de
   2 * 64 caractere hex, urmat de terminatorul null. Concatenarea se face
   prin doua bucle marcate cu #pragma unroll dupa care aplica apply_sha256 pe buffer ul concatenat si scrie rezultatul in
   vectorul de iesire.

### Functia host construct_merkle_root

Functia gestioneaza memoria si orchestreaza lansarile de kernel.

Foloseste buffere persistente. Cele trei buffere de pe gpu (d_transactions,
d_hashes_a, d_hashes_b) sunt declarate static si alocate o singura data,
la primul apel, cu dimensiunea data de max_transactions_in_a_block. Astfel se
evita un cudaMalloc si un cudaFree la fiecare bloc de tranzactii, ceea ce
ar fi introdus un overhead semnificativ deoarece functia este apelata pentru
fiecare bloc din blockchain.

Pasii executiei:

1. Copiaza tranzactiile blocului curent pe gpu printr-un singur cudaMemcpy
   host-to-device.
2. Lanseaza merkle_level0_kernel cu ceil(n / 256) block uri a cate 256 de
   thread uri.
3. Aplica reducerea iterativ. La fiecare iteratie lanseaza
   merkle_reduce_kernel, apoi interschimba buffer-ul de intrare
   cu cel de iesire. In felul acesta rezultatul unui nivel devine intrarea
   nivelului urmator fara alte copieri. Numarul de hash-uri se
   injumatateste la fiecare pas (cur_n = ceil(cur_n / 2) ).
4. Cand mai ramane un singur hash, il copiaza inapoi pe host printr-un
   cudaMemcpy device-to-host. Deoarece cudaMemcpy este sincron, el asteapta
   implicit terminarea tuturor kernel urilor lansate anterior, deci nu este
   nevoie de un cudaDeviceSynchronize explicit.


## Cautarea nonce ului

Functia find_nonce cauta cel mai mic nonce pentru care
SHA256(prev_block_hash || merkle_root || nonce), exprimat ca sir hexazecimal,
incepe cu numarul cerut de zerouri. Spatiul de cautare [0, max_nonce) este
partitionat intre thread uri.

### Observatia de optimizare: midstate caching

Prefixul asupra caruia se aplicaSHA256 este prev_block_hash || merkle_root, 
adica doua siruri hexazecimale decate 64 de caractere, in total exact 128 de bytes. 
SHA256 proceseaza mesajul in block uri interne de 64 de bytes, deci prefixul corespunde 
fix la doua transformari (sha256_transform).

Un aspect important al acestei implementari este faptul ca aceste doua transformari sunt identice pentru toate nonce urile
testate intr-un bloc, pentru ca prefixul nu se schimba. Doar ultimul block
SHA256, cel care contine cifrele nonce-ului si padding ul, difera de la un
nonce la altul.

Concluzia: in loc sa recalculam tot SHA256 ul (3 transformari) pentru fiecare
din zecile de mii de nonce uri, calculam o singura data pe host starea
intermediara dupa primii 128 de bytes si o refolosim. In
kernel, fiecare thread executa atunci o singura transformare in loc de trei.

### Pregatirea pe host (in find_nonce)

Inainte de lansarea kernel ului, host ul face trei lucruri, o singura data per
bloc:

1. Calculul midstate ului. Apeleaza  sha256_init, apoi sha256_trans pe
   primii 64 de bytes ai prefixului si inca o data pe urmatorii 64. Starea
   rezultata (SHA256_CTX.state, 8 cuvinte de 32 de biti) este copiata in
   memoria constanta c_midstate prin cudaMemcpyToSymbol.
2. Parsarea dificultatii: difficulty este primit ca sir hexazecimal de 64 de
   caractere. Este convertit in 32 de bytes raw (cu helper ul hex_nibble) si
   copiat in memoria constanta c_difficulty_bytes. Astfel comparatia din
   kernel se face pe 32 de bytes in loc de 64 de caractere hex.
3. Initializarea rezultatului. Variabila d_valid_nonce de pe gpu este setata
   la UINT32_MAX, valoare care semnaleaza faptul ca inca n am gasit nonce ul.

Memoria constanta este folosita pentru midstate si dificultate deoarece aceste
date sunt doar citite, sunt mici si identice pentru toate thread urile, caz in
care hardware-ul ofera un mecanism de broadcast foarte eficient.

### Kernel find_nonce_kernel

Sunt lansate 2048 de block uri a cate 256 de thread uri, adica 524288 de
threaduri. Fiecare thread porneste de la nonce = tid si avanseaza in pasi
de stride = gridDim.x * blockDim.x, astfel incat intreg
spatiul [0, max_nonce) este acoperit indiferent de cat de mare e max_nonce.

Pentru fiecare nonce candidat, thread ul:

1. Verifica daca valid_nonce (citit printr-un pointer
   volatile, pentru a forta recitirea din memoria globala) este deja mai mic
   sau egal cu nonce ul curent, thread ul se opreste. Motivul: cautam minimul
   global, iar toate nonce urile pe care acest thread le-ar mai testa sunt mai
   mari decat cel curent, deci nu pot imbunatati rezultatul.
2. Converteste nonce ul in sir zecimal cu intToString.
3. Construieste manual ultimul block SHA256 de 64 de bytes: cifrele nonce ului,
   urmate de byte-ul de padding 0x80, apoi zerouri, iar pe ultimii 8 bytes
   lungimea totala a mesajului in biti, in format big-endian. Lungimea totala
   este (128 + nonce_len) * 8. Aceasta constructie manuala evita overhead-ul
   functiilor sha256_update si sha256_final.
4. Initializeaza un SHA256_CTX cu starea preluata din c_midstate si apeleaza
   o singura data sha256_trans pe block ul construit.
5. Extrage hash ul final ca 32 de bytes raw direct din  ctx.state (cu
   conversie big-endian) si il compara byte cu byte cu c_difficulty_bytes.
   Comparatia se opreste la primul byte diferit: daca byte-ul hash-ului e mai
   mic, nonce-ul este valid; daca e mai mare, nu este.
6. Daca nonce-ul este valid, apeleaza atomicMin(valid_nonce, nonce).

Folosirea atomicMin garanteaza ca, indiferent de ordinea in care thread urile
gasesc nonce uri valide, in valid_nonce ramane cel mai mic dintre ele.

### Finalizarea pe host

Dupa ce kernel ul se termina (asteptat implicit de cudaMemcpy), host ul aduce
valoarea lui d_valid_nonce. Daca este inca UINT32_MAX, niciun nonce nu a
fost gasit si functia intoarce 1. Altfel, scrie nonce ul gasit in
valid_nonce si recalculeaza o singura data block_hash pe host, cu
apply_sha256, pentru a popula iesirea asteptata de restul programului.



### Prompt-uri LLM (gemini 3.5 Flash)

Ce optimizari ai face la codul din document?
Incerc sa optimizez calcularea nonce-ului pentru ca observ ca acolo se pierde destul de mult timp. Ce optimizare as putea sa fac?

Pentru a optimiza faza de **nonce finding**, care este cea mai costisitoare operație (Proof of Work), trebuie să reducem la minimum operațiile redundante din interiorul buclei `while` din kernel și să exploatăm la maximum arhitectura GPU (memorie partajată, instrucțiuni rapide intrinsic, eliminarea instrucțiunilor de salt/divergență).

Iată cele mai eficiente optimizări pe care le poți aplica direct pe codul tău:

---

## 1. Eliminarea conversiei String în kernel (Cea mai mare penalizare)

În prezent, fiecare thread apelează `intToString`, executând împărțiri (`% 10` și `/ 10`) și inversări de stringuri la fiecare iterație. Împărțirea pe întregi este extrem de lentă pe GPU (nu există hardware dedicat pentru ea, fiind emulată prin zeci de instrucțiuni).

### Optimizare: Incrementare directă în Base-10 (sau BCD) pe caractere

În loc să convertești numărul de la zero, poți menține un array de caractere local pentru fiecare thread și să îi faci incrementare directă în baza 10, similar cu un contor manual:

```cuda
// Inițializare o singură dată în afara buclei while:
char nonce_str[12];
int nonce_len = intToString((uint64_t)nonce, nonce_str);

while (nonce < max_nonce) {
    // ... folosești nonce_str ...

    // În loc de nonce += stride la final, incrementezi stringul direct cu valoarea 'stride'
    // Dacă stride este fix (de exemplu putere a lui 2), poți pre-calcula saltul direct pe caractere
}

```

*Mai bine:* Dacă `stride` este mare sau variabil, o tehnică clasică în GPU mining este ca fiecare thread să aibă o zonă de memorie tampon unde doar ultimele cifre se schimbă, sau să generezi caracterele folosind biți/baze hexazecimale dacă protocolul permite (deși de regulă se cere string decimal). Dacă se cere strict decimal, poți optimiza `intToString` folosind multiplicări cu inversul magic al lui 10 (`__umulhi`) în loc de operatorul `/`.

---

## 2. Memorarea bufferului în Shared Memory / Regiștri

În kernel, `BYTE block[64]` este alocat în memoria locală a thread-ului. Deși compilatorul încearcă să îl pună în regiștri, indexarea dinamică (ex: `block[i] = ...`) forțează GPU-ul să îl trimită în **Local Memory** (care este fizic tot în DRAM-ul global, deci are latență mare).

### Optimizare:

Deoarece lungimea prefixului este fixă (128 bytes) și stă în midstate, bufferul `block` are mereu 64 de bytes (adică exact 16 cuvinte de 32 de biți: `uint32_t W[16]`).

* În loc de `BYTE block[64]`, declară `uint32_t W[16]`.
* Mutați managementul octeților direct în acești 16 regiștri intregi prin operații pe biți (shiftări), evitând indexarea dinamică prin vectori de caractere.

---

## 3. Eliminarea Loop-ului de verificare a hash-ului

În pasul 4, ai un loop care extrage hash-ul din starea SHA și îl compară octet cu octet:

```cuda
for (int i = 0; i < 32; i++) { ... }

```

Deși are `#pragma unroll`, această comparare octet cu octet generează mult cod și salturi condiționate.

### Optimizare: Comparare pe 32/64 biți (Direct `uint32_t` sau `uint64_t`)

Deoarece datele sunt big-endian, poți converti cele 8 variabile din `ctx.state` (care sunt deja `uint32_t`) direct în formatul nativ și să le compari direct ca întregi.
Dificultatea (target-ul) are de obicei mulți de zero la început. Poți verifica direct primele elemente:

```cuda
// Re-aranjează byte-order-ul din ctx.state dacă este necesar (folosind __byte_perm)
// pentru a compara direct întregi:
uint32_t h0 = __byte_perm(ctx.state[0], 0, 0x0123); // Conversie rapidă big->little endian se face hardware

if (h0 > c_difficulty_words[0]) { valid = false; }
else if (h0 < c_difficulty_words[0]) { /* valid! */ }
else {
    // treci la următorul word doar dacă primele sunt identice
}

```

Compararea a **4 sau 8 octeți simultan** reduce drastic numărul de instrucțiuni executate.

---

## 4. Utilizarea instrucțiunilor intrinseci CUDA pentru SHA256

Dacă ai acces la implementarea funcției `sha256_transform`, asigură-te că folosește instrucțiunea intrinsecă `__byte_perm` pentru rotiri și manipulări de octeți și instrucțiunea `LOP3.LUT` (disponibilă prin compilator) pentru funcțiile logice SHA256 (`Ch`, `Maj`).

Dacă `sha256_transform` procesează datele din memoria globală sau locală, rescrie-o astfel încât să primească direct cei 16 regiștri `uint32_t W[16]` direct în hardware.

---

## 5. Unrolling și pre-calcularea padding-ului în `sha256_transform`

În mining, ultimele cuvinte din blocul de 64 de bytes (de la `W[9]` la `W[14]`) sunt **mereu 0**, iar `W[15]` conține lungimea fixă a bitlen-ului.
Dacă creezi o versiune customizată de `sha256_transform_mining`, poți elimina complet buclele din interiorul algoritmului SHA256 și să hardcodezi faptul că majoritatea elementelor din mesaj sunt zero. Acest lucru permite compilatorului să propage constantele și să elimine până la 40% din adunările din algoritmul SHA256.

---

## Rezumatul Planului de Atac

| Modificare | Impact estimat | Cum ajută |
| --- | --- | --- |
| **Înlocuire `intToString**` | **Masiv (2x - 3x)** | Elimină instrucțiunile de împărțire `/10` și `%10` de pe GPU. |
| **Comparare pe `uint32_t**` | **Mediu** | Reduce numărul de iterații de comparare de la 32 de octeți la doar 1-2 verificări de întregi. |
| **Evitarea Local Memory** | **Mediu** | Păstrarea block-ului în regiștri (`uint32_t[16]`) în loc de vector de caractere local elimină accesele ascunse la DRAM. |

Am gasit ca as putea sa fac midstate-caching. Explica-mi ce e aia

**Midstate caching** (sau stocarea stării intermediare) este o tehnică de optimizare critică în mineritul de criptomonede (cum ar fi Bitcoin) bazat pe algoritmul SHA256.

Vestea bună este că, privind codul tău, **tu ai implementat deja parțial această idee** prin intermediul constantei `c_midstate`. Totuși, haide să vedem exact ce înseamnă la nivel teoretic, de ce funcționează și cum poți împinge această optimizare și mai departe în kernel.

---

## Ce este un "Midstate"?

Algoritmul SHA256 nu procesează toate datele deodată. El sparge textul (sau blocul de date) în **bucăți fixe de exact 64 de octeți (512 biți)**.

Fiecare bucată de 64 de octeți trece printr-o funcție de compresie numită `sha256_transform`. Această funcție primește:

1. O **stare internă curentă** (formată din 8 valori de 32 de biți, inițializate cu niște constante matematice standard).
2. **Blocul curent de 64 de octeți**.

După procesarea primului bloc, starea internă se modifică. Această stare rezultată este folosită apoi ca punct de plecare pentru următorul bloc de 64 de octeți, și tot așa până la final.

> **Midstate-ul** este valoarea celor 8 regiștri interni ai SHA256 *după* ce au fost procesate primele blocuri de 64 de octeți, dar *înainte* de a fi procesat ultimul bloc (cel care conține datele care se schimbă, adică nonce-ul).

---

## De ce aduce o optimizare masivă?

În problema ta, antetul blocului (header-ul) pe care vrei să îl minezi are o structură fixă la care se adaugă la final un `nonce` variabil:

$$\text{Date finale} = \underbrace{\text{prev\_block\_hash} \ (64 \text{ bytes})}_{\text{Blocul 1}} \ \ || \ \ \underbrace{\text{merkle\_root} \ (64 \text{ bytes})}_{\text{Blocul 2}} \ \ || \ \ \underbrace{\text{nonce} \ + \ \text{padding} \ (64 \text{ bytes})}_{\text{Blocul 3}}$$

Dacă ai rula SHA256 în mod naiv pentru fiecare thread (care testează un nonce diferit):

1. Thread-ul ar procesa **Blocul 1** (64 bytes) $\rightarrow$ generează o stare intermediară.
2. Thread-ul ar procesa **Blocul 2** (64 bytes) $\rightarrow$ generează o altă stare intermediară.
3. Thread-ul ar procesa **Blocul 3** (cel cu nonce-ul) $\rightarrow$ generează hash-ul final.

Deoarece **Blocul 1** și **Blocul 2** sunt **absolut identice** pentru toate tranzacțiile din acel bloc (nu depind de nonce), este o risipă uriașă de calcul ca milioane de thread-uri GPU să repete aceleași transformări matematice de miliarde de ori.

### Soluția prin Midstate Caching:

Calculezi transformările pentru **Blocul 1** și **Blocul 2** o singură dată pe procesor (Host), obții starea de după primii 128 de octeți (midstate-ul) și o trimiți direct pe GPU.

Fiecare thread de pe GPU va sări direct la **Blocul 3**, încărcând midstate-ul în memorie și rulând o singură transformare. **Scazi volumul de muncă de la 3 transformări la 1 singură (o reducere de ~66% din timpul de calcul).**

---

## Cum este aplicat în codul tău și ce lipsește?

În funcția ta `find_nonce` din C++, ai deja implementat acest concept:

```c
// Pre-calcul pe HOST
SHA256_CTX midctx;
sha256_init(&midctx);
sha256_transform(&midctx, block_content);      // Procesează primii 64 bytes
sha256_transform(&midctx, block_content + 64); // Procesează următorii 64 bytes
// Salvezi starea în memoria constantă a GPU-ului
cudaMemcpyToSymbol(c_midstate, midctx.state, 8 * sizeof(WORD));

```

Iar în kernel, pornești direct de acolo:

```c
ctx.state[0] = c_midstate[0]; // ... inițializare rapidă
sha256_transform(&ctx, block); // O singură transformare!

```

### Unde poți optimiza și mai mult (Următorul nivel de Midstate Caching):

Deși ai implementat midstate-ul la nivel de blocuri mari, în interiorul funcției `sha256_transform` din kernel (pe care o apelezi pentru ultimul bloc de 64 de bytes), algoritmul SHA256 își extinde cele 16 cuvinte inițiale (64 bytes) într-un array de **64 de cuvinte** (mesajul extins $W$).

În cazul tău, din cei 64 de octeți ai ultimului bloc:

* Primii octeți sunt textul nonce-ului (variabil).
* Următorul octet este bitul de stop `0x80` (fix).
* Următorii octeți până la final sunt zerouri de padding (ficși).
* Ultimii 8 octeți reprezintă lungimea totală în biți (fixă).

Deoarece o mare parte din acest ultim bloc conține valori constante (`0x80`, `0x00` și lungimea), poți **pre-calcula o parte din pașii de expansiune ai mesajului ($W$)** sau poți crea o funcție specială `sha256_transform_mining` în care elimini complet operațiile matematice care adună zerouri. Multe dintre iterațiile finale ale SHA256 vor deveni simple rearanjări de regiștri, fără calcule active.


#ifndef UTILS_H#define UTILS_H#ifdef DEBUG    #define DEBUG_PRINT(...) printf(__VA_ARGS__)#else    #define DEBUG_PRINT(...)#endif#include "sha256.h"#include <stdint.h>#define SHA256_HASH_SIZE 65 // 32 byte array is 64 characters long + 1 for null terminator#define NONCE_SIZE 11 // UINT32_MAX is 10 chars long + 1 for null terminator#define BLOCK_SIZE (2 * (SHA256_HASH_SIZE - 1) + (NONCE_SIZE - 1) + 1) // Prev. block hash + Merke root Top Hash + Nonce + 1 for null terminatorvoid construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]);int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce);void warm_up_gpu();#endif // UTILS_H

te ajuta functiile din header-ul asta pentru sugestii?

Da, acest header clarifică exact constrângerile cu care lucrezi și confirmă de ce codul tău actual are anumite gâturi de gâtuire (bottlenecks) pe care le putem elimina.

Iată ce ne spune acest header și cum ne ajută să rafinăm sugestiile de optimizare:

---

### 1. Confirmarea dimensiunilor: Lucrul cu Hex String vs Raw Bytes

În header ai definit:

* `SHA256_HASH_SIZE 65` (adică 64 de caractere hexazecimale + `\0`)
* `BLOCK_SIZE` ca fiind o înșiruire de stringuri: `prev_hash (64) + merkle_root (64) + nonce (10) + \0`.

**Ce înseamnă asta pentru optimizare?**
În funcția `find_nonce_kernel` din fișierul `.cu`, tu citești `c_midstate` din memoria constantă, dar pentru că `block_content` este stocat ca text (**Hex String**), primele două blocuri au fix $64 + 64 = 128$ de caractere (bytes).

Asta înseamnă că **midstate-ul tău elimină perfect primele două blocuri de 64 de bytes**. Generarea hash-ului final din kernel se concentrează strict pe blocul 3.

---

### 2. Dimensiunea maximă a Nonce-ului (Eliminarea completă a `intToString`)

Header-ul definește `NONCE_SIZE 11` (deoarece `UINT32_MAX` adică `4294967295` are 10 cifre).

În kernelul tău curent, lungimea textului `nonce_str` variază (poate avea 1 cifră, 2 cifre... până la 10 cifre). Din această cauză, lungimea totală a mesajului se tot modifică (`nonce_len`), forțându-te să muți bitul `0x80` și să recalculezi lungimea în biți (`bitlen`) la fiecare iterație.

#### Sugestie de optimizare avansată (Padding fix la stânga):

Dacă protocolul temei tale îți permite ca textul nonce-ului să aibă o lungime fixă de 10 caractere (completat cu zerouri la stânga, ex: `0000000001` în loc de `1`), **poți elimina complet dinamismul din kernel**:

* Lungimea blocului final va fi mereu aceeași.
* Poziția bitului `0x80` va fi mereu aceeași (la indexul 10).
* `bitlen` va fi mereu o constantă fixă: $(128 + 10) \times 8 = 1104$ biți.

Dacă tema **nu** permite zerouri în stânga și cere strict formatul straturilor (ex: `"1"`, `"2"`, `"10"`), poți optimiza radical bucla prin crearea a **10 mini-bucle/mini-kerneluri sau ramificații fixe**, câte una pentru fiecare lungime de cifre (1-10), evitând astfel calculul dinamic de lungime la fiecare pas.

---

### 3. Optimizarea algoritmului `sha256_transform` din interior

Știind că dimensiunea maximă a string-ului nonce este de 10 caractere, înseamnă că în `BYTE block[64]`:

* `block[0]` până la `block[9]` reprezintă caracterele nonce-ului.
* `block[10]` până la `block[55]` sunt, în cel mai rău caz, în mare parte **zerouri** (în afară de acel singur byte `0x80`).
* `block[56]` până la `block[63]` conțin lungimea în biți.

Când acest bloc este mapat în cele 16 cuvinte de 32 de biți (`W[0]` ... `W[15]`) pentru SHA256:

* Doar `W[0]`, `W[1]` și `W[2]` conțin date dinamice din nonce.
* `W[3]` până la `W[13]` sunt **complet zero**.
* `W[14]` și `W[15]` conțin bitlen-ul (valori aproape constante).

**Cum te ajută asta?**
Dacă rescrii funcția `sha256_transform` special pentru acest kernel (o poți numi `sha256_transform_mining`), poți să ștergi faza de extindere a mesajului (bucla care generează de la `W[16]` la `W[64]`) și să o înlocuiești cu ecuații pre-calculate în care majoritatea termenilor fiind `0` dispar. Compilatorul NVCC va optimiza masiv regiștrii, rezultând într-o execuție mult mai rapidă per thread.

---

### Cum ar arăta structura optimizată a kernelului în regiștri (fără vectori de caractere)

Folosind constantele din header, putem schimba structura kernelului ca să lucreze direct pe întregi (`uint32_t`), eliminând complet `char nonce_str[12]` și `BYTE block[64]`:

```cuda
__global__ void find_nonce_kernel_optimized(uint32_t max_nonce, uint32_t *valid_nonce) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    volatile uint32_t *vn = valid_nonce;

    uint32_t nonce = tid;
    while (nonce < max_nonce) {
        if (*vn <= nonce) return;

        // În loc de block[64], definim direct starea mesajului SHA256 (16 word-uri)
        uint32_t W[16] = {0};

        // 1. Convertim nonce-ul direct în octeți ASCII direct în W[0], W[1], W[2] 
        // folosind operații rapide de shiftare în loc de intToString.
        // (De exemplu, extragi cifrele prin înmulțire cu inversul magic și le pui în regiștri)
        
        // 2. Hardcodăm padding-ul direct în regiștri:
        // Presupunând că nonce-ul are 10 cifre fix (W[0], W[1] și jumătate din W[2]):
        // W[2] |= 0x80000000; // Pune bitul de stop imediat după cele 10 caractere
        W[15] = 1104;         // Lungimea fixă în biți: (128 + 10) * 8

        // 3. Apelăm transformarea customizată care primește direct regiștrii
        SHA256_CTX ctx;
        ctx.state[0] = c_midstate[0]; // ... inițializare din constant memory
        
        sha256_transform_registers(&ctx, W); 

        // 4. Comparare rapidă pe unit32_t
        // Nu mai transformi în hex string! Compari direct ctx.state[0] cu c_difficulty_words[0]
        if (ctx.state[0] < c_difficulty_words[0]) {
            atomicMin(valid_nonce, nonce);
            return;
        }
        
        nonce += stride;
    }
}

```