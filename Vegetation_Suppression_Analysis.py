import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

class VegetationAnalyzer:
    def __init__(self):
        """
        Inicializa o analisador de vegetação
        """
        self.green_threshold_hsv = {
            'lower': np.array([35, 40, 40]),   # Limite inferior HSV para verde
            'upper': np.array([85, 255, 255]) # Limite superior HSV para verde
        }
        
    def load_image(self, image_path):
        """
        Carrega e pré-processa a imagem
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Não foi possível carregar a imagem: {image_path}")
            
            # Converte BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        except Exception as e:
            print(f"Erro ao carregar imagem: {e}")
            return None
    
    def preprocess_image(self, image):
        """
        Pré-processa a imagem para melhorar a detecção de vegetação
        """
        # Aplica filtro bilateral para reduzir ruído mantendo bordas
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Converte para HSV para melhor separação de cores
        hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
        
        return hsv
    
    def detect_vegetation_hsv(self, image):
        """
        Detecta vegetação usando espaço de cores HSV
        """
        hsv = self.preprocess_image(image)
        
        # Cria máscara para tons de verde
        mask = cv2.inRange(hsv, self.green_threshold_hsv['lower'], 
                          self.green_threshold_hsv['upper'])
        
        # Aplica operações morfológicas para limpar a máscara
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    
    def detect_vegetation_ndvi(self, image):
        """
        Simula cálculo NDVI usando canais RGB
        (Aproximação - NDVI real requer bandas NIR)
        """
        # Converte para float
        img_float = image.astype(np.float64)
        
        # Simula NDVI usando (G - R) / (G + R)
        red = img_float[:, :, 0]
        green = img_float[:, :, 1]
        
        # Evita divisão por zero
        denominator = green + red
        denominator[denominator == 0] = 1
        
        ndvi = (green - red) / denominator
        
        # Normaliza NDVI para 0-255
        ndvi_normalized = ((ndvi + 1) * 127.5).astype(np.uint8)
        
        # Threshold para vegetação (NDVI > 0.2)
        vegetation_mask = (ndvi > 0.2).astype(np.uint8) * 255
        
        return vegetation_mask
    
    def detect_vegetation_kmeans(self, image, n_clusters=8):
        """
        Usa K-means clustering para identificar vegetação
        """
        # Reshape da imagem para clustering
        pixels = image.reshape(-1, 3)
        
        # Aplica K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Identifica clusters que representam vegetação
        cluster_centers = kmeans.cluster_centers_
        
        # Identifica clusters verdes baseado na componente verde
        green_clusters = []
        for i, center in enumerate(cluster_centers):
            # Converte para HSV para análise
            center_hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_RGB2HSV)[0][0]
            
            # Verifica se está na faixa de verde
            if (35 <= center_hsv[0] <= 85 and 
                center_hsv[1] > 30 and 
                center_hsv[2] > 30):
                green_clusters.append(i)
        
        # Cria máscara para clusters verdes
        vegetation_mask = np.zeros(labels.shape, dtype=np.uint8)
        for cluster in green_clusters:
            vegetation_mask[labels == cluster] = 255
        
        return vegetation_mask.reshape(image.shape[:2])
    
    def combine_methods(self, image):
        """
        Combina diferentes métodos para detecção mais robusta
        """
        mask_hsv = self.detect_vegetation_hsv(image)
        mask_ndvi = self.detect_vegetation_ndvi(image)
        mask_kmeans = self.detect_vegetation_kmeans(image)
        
        # Combina máscaras usando operação OR
        combined_mask = cv2.bitwise_or(mask_hsv, mask_ndvi)
        combined_mask = cv2.bitwise_or(combined_mask, mask_kmeans)
        
        # Aplica filtro de mediana para reduzir ruído
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        return combined_mask, mask_hsv, mask_ndvi, mask_kmeans
    
    def calculate_vegetation_area(self, mask):
        """
        Calcula a área de vegetação em pixels e porcentagem
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        vegetation_pixels = np.sum(mask > 0)
        vegetation_percentage = (vegetation_pixels / total_pixels) * 100
        
        return vegetation_pixels, vegetation_percentage
    
    def analyze_images(self, before_path, after_path):
        """
        Analisa ambas as imagens e calcula a supressão
        """
        print("Carregando imagens...")
        
        # Carrega as imagens
        image_before = self.load_image(before_path)
        image_after = self.load_image(after_path)
        
        if image_before is None or image_after is None:
            return None
        
        print("Detectando vegetação na imagem 'antes'...")
        mask_before, hsv_before, ndvi_before, kmeans_before = self.combine_methods(image_before)
        
        print("Detectando vegetação na imagem 'depois'...")
        mask_after, hsv_after, ndvi_after, kmeans_after = self.combine_methods(image_after)
        
        # Calcula áreas de vegetação
        pixels_before, percent_before = self.calculate_vegetation_area(mask_before)
        pixels_after, percent_after = self.calculate_vegetation_area(mask_after)
        
        # Calcula supressão
        suppression_pixels = pixels_before - pixels_after
        suppression_percentage = ((suppression_pixels / pixels_before) * 100) if pixels_before > 0 else 0
        
        # Resultados
        results = {
            'before': {
                'pixels': pixels_before,
                'percentage': percent_before,
                'mask': mask_before
            },
            'after': {
                'pixels': pixels_after,
                'percentage': percent_after,
                'mask': mask_after
            },
            'suppression': {
                'pixels': suppression_pixels,
                'percentage': suppression_percentage,
                'absolute_reduction': percent_before - percent_after
            }
        }
        
        return results, image_before, image_after
    
    def visualize_results(self, results, image_before, image_after):
        """
        Visualiza os resultados da análise
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Imagem original - antes
        axes[0, 0].imshow(image_before)
        axes[0, 0].set_title('Gaza - Antes da Guerra')
        axes[0, 0].axis('off')
        
        # Máscara de vegetação - antes
        axes[0, 1].imshow(results['before']['mask'], cmap='Greens')
        axes[0, 1].set_title(f'Vegetação Detectada - Antes\n{results["before"]["percentage"]:.2f}% da área')
        axes[0, 1].axis('off')
        
        # Sobreposição - antes
        overlay_before = image_before.copy()
        overlay_before[results['before']['mask'] > 0] = [0, 255, 0]  # Verde
        axes[0, 2].imshow(cv2.addWeighted(image_before, 0.7, overlay_before, 0.3, 0))
        axes[0, 2].set_title('Sobreposição - Antes')
        axes[0, 2].axis('off')
        
        # Imagem original - depois
        axes[1, 0].imshow(image_after)
        axes[1, 0].set_title('Gaza - Depois da Guerra')
        axes[1, 0].axis('off')
        
        # Máscara de vegetação - depois
        axes[1, 1].imshow(results['after']['mask'], cmap='Greens')
        axes[1, 1].set_title(f'Vegetação Detectada - Depois\n{results["after"]["percentage"]:.2f}% da área')
        axes[1, 1].axis('off')
        
        # Sobreposição - depois
        overlay_after = image_after.copy()
        overlay_after[results['after']['mask'] > 0] = [0, 255, 0]  # Verde
        axes[1, 2].imshow(cv2.addWeighted(image_after, 0.7, overlay_after, 0.3, 0))
        axes[1, 2].set_title('Sobreposição - Depois')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Análise de Supressão Vegetal em Gaza\n'
                    f'Supressão: {results["suppression"]["percentage"]:.1f}% '
                    f'({results["suppression"]["absolute_reduction"]:.2f} pontos percentuais)',
                    fontsize=16, y=1.02)
        plt.show()
        
        return fig
    
    def generate_report(self, results):
        """
        Gera relatório detalhado da análise
        """
        report = f"""
RELATÓRIO DE ANÁLISE DE SUPRESSÃO VEGETAL - GAZA
{'='*50}

RESUMO EXECUTIVO:
- Área vegetal antes da guerra: {results['before']['percentage']:.2f}% da área total
- Área vegetal após a guerra: {results['after']['percentage']:.2f}% da área total
- Supressão total: {results['suppression']['percentage']:.1f}%
- Redução absoluta: {results['suppression']['absolute_reduction']:.2f} pontos percentuais

DETALHES TÉCNICOS:
- Pixels de vegetação (antes): {results['before']['pixels']:,}
- Pixels de vegetação (depois): {results['after']['pixels']:,}
- Pixels suprimidos: {results['suppression']['pixels']:,}

METODOLOGIA:
O algoritmo utiliza três técnicas complementares:
1. Análise HSV: Detecção baseada em matiz, saturação e valor
2. NDVI Simulado: Índice de vegetação usando canais RGB
3. K-means Clustering: Agrupamento de cores similares

INTERPRETAÇÃO:
A análise indica uma {'severa' if results['suppression']['percentage'] > 50 else 'significativa' if results['suppression']['percentage'] > 25 else 'moderada'} 
supressão da cobertura vegetal na região analisada.
        """
        
        return report

# Exemplo de uso
def main():
    # Inicializa o analisador
    analyzer = VegetationAnalyzer()
    
    # Caminhos das imagens (ajuste conforme necessário)
    before_image = "gaza_before.jpg"  # Substitua pelo caminho real
    after_image = "gaza_after.jpg"   # Substitua pelo caminho real
    
    print("Iniciando análise de supressão vegetal em Gaza...")
    print("="*50)
    
    # Analisa as imagens
    results, img_before, img_after = analyzer.analyze_images(before_image, after_image)
    
    if results:
        # Gera e exibe relatório
        report = analyzer.generate_report(results)
        print(report)
        
        # Visualiza resultados
        analyzer.visualize_results(results, img_before, img_after)
    else:
        print("Erro na análise. Verifique os caminhos das imagens.")

if __name__ == "__main__":
    main()
