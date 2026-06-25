#!/bin/bash
#SBATCH --job-name=DeepLMI_Train         # İşin adı
#SBATCH --output=deeplmi_train_%j.out    # Standart çıktı dosyası (%j job ID'yi temsil eder)
#SBATCH --error=deeplmi_train_%j.err     # Hata çıktı dosyası
#SBATCH --time=48:00:00                  # Maksimum çalışma süresi (SS:DD:AA - 48 saat)
#SBATCH --nodes=1                        # Kullanılacak node (sunucu) sayısı
#SBATCH --ntasks=1                       # Çalıştırılacak görev sayısı
#SBATCH --cpus-per-task=8                # Veri ön işleme için CPU çekirdek sayısı
#SBATCH --mem=64G                        # İstenen RAM miktarı (Graph yapısı ve Doc2Vec için bol RAM iyidir)
#SBATCH --partition=gpu                  # GPU partition (kuyruk) adı (sunucunuza göre değişebilir)
#SBATCH --gres=gpu:1                     # 1 adet GPU talep ediliyor

# Çevresel değişkenleri yükle ve conda ortamını aktif et
# Eğer Miniconda kullanıyorsanız yolu kontrol edin (örn: ~/miniconda3/etc/profile.d/conda.sh)
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh

conda activate deeplmi

# Çalışma dizinine git (SLURM default olarak komutun verildiği dizinde başlar ama emin olmak için)
# cd /yol/neredeyse/DeepLMI_klasoru

echo "=========================================================="
echo "İşlem başlangıç zamanı: $(date)"
echo "Çalıştığı Node: $SLURMD_NODENAME"
echo "Kullanılan GPU'lar: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Eğitim scriptini çalıştır
python main_cold_start_custom.py

echo "=========================================================="
echo "İşlem bitiş zamanı: $(date)"
echo "=========================================================="
