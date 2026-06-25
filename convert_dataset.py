import os
import json
import pandas as pd

def process_custom_dataset(training_chunks_dir, data_with_negatives_dir, fasta_files_dir, output_dir="dataset_custom"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Filtreleme: Sadece miRNA-lncRNA etkileşimlerini al
    print("Veriler filtreleniyor...")
    
    def extract_mirna_lncrna(jsonl_dir):
        interactions = []
        unique_mirnas = set()
        unique_lncrnas = set()
        
        for file in os.listdir(jsonl_dir):
            if not file.endswith(".jsonl"): continue
            filepath = os.path.join(jsonl_dir, file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Sadece RNA-RNA ve miRNA-lncRNA etkileşimlerini alıyoruz
                    if data.get("interaction_type") == "rna-rna":
                        rna_type = data.get("RNA_type", "")
                        target_type = data.get("target_RNA_type", "")
                        
                        if (rna_type == "miRNA" and target_type == "lncRNA") or \
                           (rna_type == "lncRNA" and target_type == "miRNA"):
                            
                            mirna_id = data["RNA_id"] if rna_type == "miRNA" else data["target_id"]
                            lncrna_id = data["target_id"] if rna_type == "miRNA" else data["RNA_id"]
                            label = data["interaction_label"]
                            
                            unique_mirnas.add(mirna_id)
                            unique_lncrnas.add(lncrna_id)
                            interactions.append((mirna_id, lncrna_id, label))
                            
        return interactions, unique_mirnas, unique_lncrnas

    train_interactions, train_mirnas, train_lncrnas = extract_mirna_lncrna(training_chunks_dir)
    test_interactions, test_mirnas, test_lncrnas = extract_mirna_lncrna(data_with_negatives_dir)
    
    all_mirnas = list(train_mirnas.union(test_mirnas))
    all_lncrnas = list(train_lncrnas.union(test_lncrnas))
    
    print(f"Toplam miRNA sayısı: {len(all_mirnas)}, Toplam lncRNA sayısı: {len(all_lncrnas)}")
    print(f"Eğitim etkileşim sayısı: {len(train_interactions)}, Test etkileşim sayısı: {len(test_interactions)}")
    
    # 2. Index Oluşturma (index_value.csv)
    # DeepLMI tüm RNA'lar için ardışık bir ID bekler
    rna_to_index = {}
    index_records = []
    
    current_idx = 0
    for mirna in all_mirnas:
        rna_to_index[mirna] = current_idx
        index_records.append({"rna": mirna, "index": current_idx})
        current_idx += 1
        
    for lncrna in all_lncrnas:
        rna_to_index[lncrna] = current_idx
        index_records.append({"rna": lncrna, "index": current_idx})
        current_idx += 1
        
    pd.DataFrame(index_records).to_csv(os.path.join(output_dir, "index_value.csv"), index=False)
    
    # 3. Node Link dosyalarını oluşturma
    def create_node_link(interactions, out_name):
        records = []
        for mirna, lncrna, label in interactions:
            records.append({
                "node1": rna_to_index[mirna],
                "node2": rna_to_index[lncrna],
                "label": label
            })
        pd.DataFrame(records).to_csv(os.path.join(output_dir, out_name), index=False)
        
    create_node_link(train_interactions, "node_link_train.csv")
    create_node_link(test_interactions, "node_link_testval.csv")
    
    # 4. İlgili FASTA dizilimlerini bulma ve kaydetme
    print("FASTA dosyaları oluşturuluyor (Büyük rna.fa dosyası taranıyor, bu işlem biraz sürebilir)...")
    
    mirna_set = set(all_mirnas)
    lncrna_set = set(all_lncrnas)
    
    found_mirnas = set()
    found_lncrnas = set()
    
    mirna_out_path = os.path.join(output_dir, "mirna.fasta")
    lncrna_out_path = os.path.join(output_dir, "lncrna.fasta")
    
    with open(mirna_out_path, "w", encoding="utf-8") as f_mirna, \
         open(lncrna_out_path, "w", encoding="utf-8") as f_lncrna:
        
        # Sadece rna.fa dosyasını okuyoruz
        rna_fa_path = os.path.join(fasta_files_dir, "rna.fa")
        if os.path.exists(rna_fa_path):
            with open(rna_fa_path, "r", encoding="utf-8") as fa:
                write_to = None
                for line in fa:
                    if line.startswith(">"):
                        full_id = line[1:].strip()
                        short_id = full_id.split()[0]
                        
                        if full_id in mirna_set or short_id in mirna_set:
                            write_to = f_mirna
                            current_id = full_id if full_id in mirna_set else short_id
                            found_mirnas.add(current_id)
                            write_to.write(f">{current_id}\n")
                        elif full_id in lncrna_set or short_id in lncrna_set:
                            write_to = f_lncrna
                            current_id = full_id if full_id in lncrna_set else short_id
                            found_lncrnas.add(current_id)
                            write_to.write(f">{current_id}\n")
                        else:
                            write_to = None
                    else:
                        if write_to is not None:
                            write_to.write(line)
        else:
            print(f"UYARI: {rna_fa_path} dosyası bulunamadı!")
            
    print(f"Çıkarılan miRNA sayısı: {len(found_mirnas)} / {len(mirna_set)}")
    print(f"Çıkarılan lncRNA sayısı: {len(found_lncrnas)} / {len(lncrna_set)}")
    
    missing_mirnas = mirna_set - found_mirnas
    missing_lncrnas = lncrna_set - found_lncrnas
    
    with open(mirna_out_path, "a", encoding="utf-8") as f_mirna:
        for m in missing_mirnas:
            f_mirna.write(f">{m}\nN\n")
            
    with open(lncrna_out_path, "a", encoding="utf-8") as f_lncrna:
        for m in missing_lncrnas:
            f_lncrna.write(f">{m}\nN\n")

    if missing_mirnas or missing_lncrnas:
        print("UYARI: Bazı RNA'lar FASTA dosyasında bulunamadı. Hata vermemesi için onlara 'N' (bilinmeyen) dizilimi atandı.")

    print(f"\nDönüşüm tamamlandı! Çıktılar {output_dir} klasöründe.")
    print("Sıradaki adımlar: ")
    print("1. SPOT-RNA-2D ile mirna.fasta dosyası üzerinden yapıları çıkarıp mirna_contact klasörüne koyun.")
    print("2. RNA-FM ile mirna.fasta üzerinden embeddingleri çıkarıp mirna_emb klasörüne koyun.")

if __name__ == "__main__":
    # Yolları veri setinizin gerçek konumuna göre güncelleyin
    training_chunks = "training_chunks"
    data_with_negatives = "data_with_negatives" # Örnek yol
    fasta_files = "fasta_files" # Örnek yol
    
    process_custom_dataset(training_chunks, data_with_negatives, fasta_files)
