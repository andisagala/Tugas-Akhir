# Tugas Akhir: Hand Landmark Detection & Android Integration

Tugas akhir ini terdiri dari dua bagian:

- **prepare model**: Berisi kode Python untuk melakukan preprocessing dataset hingga menghasilkan model `.tflite` untnuk digunakan pada aplikasi Android.
- **Android Apps**: Berisi source code aplikasi Android.

## Struktur Folder `prepare model`

```
prepare model/
├── Dataset/
│   ├── Label 1/
│   ├── Label 2/
│   ├── ...
│   └── Label ke-21/
├── dataprep.ipynb
├── training.ipynb
├── converth5totflite.py
└── realing.py
```

### Alur Pengolahan Data dan Pembuatan Model

1. **Persiapan Dataset**  
   Dataset gambar diletakkan dalam folder `Dataset/`, dengan masing-masing label/kelas memiliki subfolder tersendiri (misal: Label 1, Label 2, ..., Label ke-21).

2. **Ekstraksi Fitur & Pembuatan Dataset TensorFlow**  
   Jalankan `dataprep.ipynb` untuk memproses gambar-gambar di folder `Dataset/`. Notebook ini akan mengekstrak koordinat hasil hand landmarking dari setiap gambar dan menghasilkan file dataset TensorFlow berformat `.tfrecords`.

3. **Training Model**  
   Gunakan notebook `training.ipynb` untuk melatih model menggunakan file `.tfrecords` yang telah dibuat. Hasil pelatihan berupa model TensorFlow dengan ekstensi `.h5`.

4. **Konversi Model ke TFLite**  
   Model `.h5` kemudian dikonversi menjadi format `.tflite` menggunakan script `converth5totflite.py`, sehingga dapat digunakan pada aplikasi Android.

5. **Pengujian Model di Komputer**  
   Untuk menguji performa model pada komputer, gunakan script `realing.py`.

6. **Integrasi ke Android**  
   Jika model sudah dianggap cukup baik, file `.tflite` dapat diintegrasikan ke dalam aplikasi Android yang terdapat pada folder `Android Apps`.

---
