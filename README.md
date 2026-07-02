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
    }
}

```
