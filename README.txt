======== READ ME =============

Autrices :

CHAABOUNI Sarah (aka : Anonymous 2)
TROUBAT Victoria (aka : Anonymous 1)

==== Mise en contexte =====

Le but de ce code est d'homogénéiser une image à l'aide d'un GPU et CPU.
Pour cela vous trouverez dans ce dossier le code en C puis en Cuda pour le GPU.
Vous y trouverez l'image "lena_gray.bmp" pour l'image d'entrée.
Nous avons laissé l'image de sortie qui s'appelle "leanGPU2.bmp"

==== Executer le code =====

Pour faire tourner ce code, il vous faudra vous munir d'un GPU. Nous avons utilisé le serveur runpod.io.
prenez un pod qui contient cuda.
allez dans settings et connectez vous à github.
lorsque vous etes connectés au pod entrez dans le terminal : 
- git clone git@github.com:sasachaa/Heterogeneous-Programming.git
- cd Heterogeneous-Programming/
- nvcc -o bilat2 filtreBilat.cu 
- ./bilat2 lena_gray.bmp leanGPU3.bmp

