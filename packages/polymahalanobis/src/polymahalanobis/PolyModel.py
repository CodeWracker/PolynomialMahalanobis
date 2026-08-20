from __future__ import annotations

import numpy as np
import logging
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)


class LevelBasis:
    """Representa um nível da expansão polinomial (equivalente ao lev_basis do C++)"""
    def __init__(self) -> None:
        self.A_basis: np.ndarray | None = None      # Matriz de base (Ucont no C++)
        self.max_aP: float = 1.0        # Valor máximo de normalização
        self.sigma_inv: float = 1.0     # Inverso de sigma (1/s_min)
        self.dms: np.ndarray | None = None          # Pesos especiais (array)
        self.ind_use: np.ndarray | None = None      # Índices das dimensões usadas
        self.d_proj: int = 0          # Número de dimensões projetadas


class PolyMahalanobis:
    """
    Implementação do Polynomial Mahalanobis Distance Classifier
    Suporta dados multiespectrais com número arbitrário de bandas.
    Segue exatamente a lógica do código C++ original.
    """
    
    def __init__(self, sample_file: str, num_levels: int = 3) -> None:
        """
        Inicializa o classificador
        
        Args:
            sample_file: caminho para arquivo com amostras (formato texto, uma amostra por linha)
                        Cada linha deve ter N valores (N = número de bandas)
            num_levels: número de níveis polinomiais (padrão 3)
        """
        self.num_levels: int = num_levels
        self.eps_svd: float = 4e-6  # CRÍTICO: mesmo valor do C++ (sig_max)
        self.samples: np.ndarray = self._load_samples(sample_file)
        self.n_bands: int = self.samples.shape[1]
        self.center: np.ndarray | None = None
        self.levels: list[LevelBasis] = []
        
    def _load_samples(self, sample_file: str) -> np.ndarray:
        """Carrega amostras do arquivo"""
        logger.info(f"Loading samples from {sample_file}")
        samples = np.loadtxt(sample_file, dtype=np.float32)
        logger.info(f"Loaded {samples.shape[0]} samples with {samples.shape[1]} bands")
        return samples
        
    def makeSpace(self) -> None:
        """
        Cria o espaço topológico (equivalente ao makeSpace do C++)
        Este é o método de treinamento principal
        """
        logger.info("Creating topological space")
        
        # Calcula centro (média das amostras)
        self.center = np.mean(self.samples, axis=0)
        logger.info(f"Center: {self.center}")
        
        # Centraliza dados: A = X - mean(X)
        A = self.samples - self.center
        nt, dt = A.shape
        logger.info(f"Centralized data shape: ({nt}, {dt})")
        
        # ========== PRIMEIRO NÍVEL ==========
        logger.info("Computing level 1...")
        level = self._compute_svd_level(A, nt, dt)
        self.levels.append(level)
        
        # Projeção inicial: proj_A = A * Ucont
        proj_A = A @ level.A_basis
        proj_A, level.max_aP = self._normalize_projection(proj_A)
        
        n_proj, d_proj = proj_A.shape
        level.d_proj = d_proj
        logger.info(f"Level 1: d_proj={d_proj}, max_aP={level.max_aP:.6f}")
        
        # Verifica se deve continuar expandindo
        cont = self._should_continue(proj_A, d_proj)
        
        if self.num_levels == 1:
            cont = False
            
        # ========== NÍVEIS SUBSEQUENTES ==========
        level_count = 1
        while cont and level_count < self.num_levels:
            logger.info(f"Computing level {level_count + 1}...")
            
            # Cria nova dimensão com termos polinomiais
            new_dim = self._create_polynomial_expansion(proj_A, d_proj)
            logger.info(f"Polynomial expansion created: {new_dim.shape}")
            
            # Remove dimensões com variância muito baixa
            A_next, ind_use = self._remove_low_variance_dims(new_dim)
            logger.info(f"After variance filtering: {A_next.shape}, kept {len(ind_use)} dimensions")
            
            # Computa SVD no novo espaço
            nt, dt = A_next.shape
            level = self._compute_svd_level(A_next, nt, dt)
            level.ind_use = ind_use
            self.levels.append(level)
            
            # Projeção
            proj_A = A_next @ level.A_basis
            proj_A, level.max_aP = self._normalize_projection(proj_A)
            
            n_proj, d_proj = proj_A.shape
            level.d_proj = d_proj
            logger.info(f"Level {level_count + 1}: d_proj={d_proj}, max_aP={level.max_aP:.6f}")
            
            # Verifica se deve continuar
            cont = self._should_continue(proj_A, d_proj)
            level_count += 1
            
        logger.info(f"Topological space created with {len(self.levels)} levels")
        self._log_levels_info()
    
    def _compute_svd_level(self, A: np.ndarray, nt: int, dt: int) -> LevelBasis:
        """
        Computa SVD e cria um nível (replica a lógica do C++)
        
        Existem dois casos:
        1. nt >= dt: usa A'*A (mais amostras que dimensões)
        2. nt < dt: usa A*A' (mais dimensões que amostras) + normalização especial
        """
        level = LevelBasis()
        
        if nt >= dt:
            # CASO 1: Mais amostras que dimensões
            # AtA = A'*A
            AtA = A.T @ A
            U, S, Vt = np.linalg.svd(AtA, full_matrices=True)
            s_val = S
            
            # Filtragem de valores singulares
            s_max = np.max(s_val)
            s_min = self.eps_svd * s_max  # CRÍTICO: threshold
            
            # Índices dos valores singulares válidos (> s_min)
            ind_basis = np.where(s_val > s_min)[0]
            
            if len(ind_basis) == 0:
                logger.warning("No valid singular values found! Using first component.")
                ind_basis = np.array([0])
            
            # Autovalores e base
            my_lambda = s_val[ind_basis]
            Ucont = U[:, ind_basis]
            
        else:
            # CASO 2: Mais dimensões que amostras
            # AAt = A*A'
            AAt = A @ A.T
            U, S, Vt = np.linalg.svd(AAt, full_matrices=True)
            s_val = S
            
            # Filtragem
            s_max = np.max(s_val)
            s_min = self.eps_svd * s_max
            
            ind_basis = np.where(s_val > s_min)[0]
            
            if len(ind_basis) == 0:
                logger.warning("No valid singular values found! Using first component.")
                ind_basis = np.array([0])
                
            my_lambda = s_val[ind_basis]
            U_basis = U[:, ind_basis]
            
            # CRÍTICO: Projeção e normalização por coluna
            # Ucont = (U' * A)'
            Ucont = U_basis.T @ A  # Shape: (n_components, d)
            Ucont = Ucont.T        # Shape: (d, n_components)
            
            # NORMALIZAÇÃO POR COLUNA (essencial para consistência!)
            # Ucont_dist = sqrt(sum(Ucont.^2))
            # Ucont = Ucont / Ucont_dist
            Ucont_dist = np.sqrt(np.sum(Ucont**2, axis=0))
            Ucont = Ucont / (Ucont_dist + 1e-10)  # Evita divisão por zero
            
            s_min = self.eps_svd * s_max
        
        # Armazena no nível
        level.A_basis = Ucont.astype(np.float32)
        level.sigma_inv = 1.0 / s_min
        
        # Calcula DMS (pesos especiais)
        # dms = -lambda / (s_min * (lambda + s_min))
        level.dms = (-my_lambda / (s_min * (my_lambda + s_min))).astype(np.float32)
        
        logger.debug(f"SVD level computed: A_basis shape={level.A_basis.shape}, "
                    f"sigma_inv={level.sigma_inv:.6e}, dms shape={level.dms.shape}")
        
        return level
    
    def _normalize_projection(self, proj_A: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Normaliza projeção pelo valor absoluto máximo
        if max_aP > eps_svd: proj_A = proj_A / max_aP
        """
        max_aP = np.max(np.abs(proj_A))
        if max_aP > self.eps_svd:
            proj_A = proj_A / max_aP
        else:
            max_aP = 1.0
        return proj_A, max_aP
    
    def _create_polynomial_expansion(self, proj_A: np.ndarray, d_proj: int) -> np.ndarray:
        """
        Cria expansão polinomial customizada
        new_dim = [proj_A, proj_A.^2, cross_terms]
        
        Os termos cruzados são produtos de pares de dimensões: proj_A[:, i] * proj_A[:, j]
        """
        if d_proj > 1:
            # Termos quadráticos
            proj_A_sq = proj_A ** 2
            
            # Termos cruzados: todas as combinações de 2 dimensões
            # C = nchoosek(1:d_proj, 2)
            cross_terms_list = []
            for i, j in combinations(range(d_proj), 2):
                cross_terms_list.append(proj_A[:, i] * proj_A[:, j])
            
            if cross_terms_list:
                cross_terms = np.column_stack(cross_terms_list)
            else:
                cross_terms = np.array([]).reshape(proj_A.shape[0], 0)
            
            # Concatena: [proj_A, proj_A^2, cross_terms]
            new_dim = np.hstack([proj_A, proj_A_sq, cross_terms])
        else:
            # Se d_proj == 1, apenas [proj_A, proj_A^2]
            new_dim = np.hstack([proj_A, proj_A ** 2])
            
        return new_dim.astype(np.float32)
    
    def _remove_low_variance_dims(self, new_dim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove dimensões com variância muito baixa
        ind_use = find(var_new_dim > 1e-8 * max(var_new_dim))
        """
        var_new_dim = np.var(new_dim, axis=0)
        max_var = np.max(var_new_dim)
        
        # Threshold: 1e-8 * max_var
        ind_use = np.where(var_new_dim > 1e-8 * max_var)[0]
        
        if len(ind_use) == 0:
            # Fallback: mantém pelo menos a primeira dimensão
            logger.warning("All dimensions filtered out by variance! Keeping first dimension.")
            ind_use = np.array([0])
            
        A_next = new_dim[:, ind_use]
        return A_next, ind_use
    
    def _should_continue(self, proj_A: np.ndarray, d_proj: int) -> bool:
        """
        Verifica se deve continuar a expansão polinomial
        Critério: variância da nova dimensão > eps_svd
        """
        if d_proj <= 1:
            return False
            
        # Cria expansão temporária para verificar variância
        new_dim = self._create_polynomial_expansion(proj_A, d_proj)
        var = np.var(new_dim)
        
        return var > self.eps_svd
    
    def evaluate(self, spectral_values: np.ndarray) -> np.ndarray:
        """
        Avalia distâncias polinomiais para valores espectrais
        
        Args:
            spectral_values: array (N, n_bands) ou (n_bands,) com valores das bandas
            
        Returns:
            array (N, num_levels) com distâncias acumuladas por nível
        """
        if spectral_values.ndim == 1:
            spectral_values = spectral_values.reshape(1, -1)
        
        # Valida número de bandas
        if spectral_values.shape[1] != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, got {spectral_values.shape[1]}")
            
        N = spectral_values.shape[0]
        
        # CRÍTICO: Centraliza os dados de entrada!
        # X = U - center
        X = (spectral_values - self.center).astype(np.float32)
        
        distances = np.zeros((N, len(self.levels)), dtype=np.float32)
        
        # ========== PRIMEIRO NÍVEL ==========
        level = self.levels[0]
        
        # proj_A = X * A_basis
        proj_A = X @ level.A_basis
        proj_A_sq = proj_A ** 2
        A_sq = X ** 2
        
        # Calcula q1 e q2 (fórmula específica do C++)
        # q1 = sum((A_sq * sigma_inv)')'
        # q2 = sum((proj_A_sq .* dms)')'
        q1 = np.sum(A_sq * level.sigma_inv, axis=1)
        q2 = np.sum(proj_A_sq * level.dms, axis=1)
        
        # q_in = q1 + q2
        # q_in(q_in < 0) = 0
        q_in = q1 + q2
        q_in = np.maximum(q_in, 0)  # Clipping para não-negativos
        
        distances[:, 0] = q_in
        
        # ========== NÍVEIS SUBSEQUENTES ==========
        if len(self.levels) > 1:
            # Prepara new_dim para próximo nível
            proj_A_normalized = proj_A / level.max_aP
            new_dim = self._create_polynomial_expansion(proj_A_normalized, level.d_proj)
            
            for level_idx in range(1, len(self.levels)):
                level = self.levels[level_idx]
                
                # Seleciona apenas dimensões usadas (ind_use)
                new_dim_used = new_dim[:, level.ind_use]
                
                # Projeção
                proj_A = new_dim_used @ level.A_basis
                proj_A_sq = proj_A ** 2
                A_sq = new_dim_used ** 2
                
                # q1 e q2
                q1 = np.sum(A_sq * level.sigma_inv, axis=1)
                q2 = np.sum(proj_A_sq * level.dms, axis=1)
                
                q_in = q1 + q2
                q_in = np.maximum(q_in, 0)
                
                # CRÍTICO: Acumula com nível anterior!
                # distances[:, level] = q_in + distances[:, level-1]
                distances[:, level_idx] = q_in + distances[:, level_idx - 1]
                
                # Prepara para próximo nível (se houver)
                if level_idx < len(self.levels) - 1:
                    proj_A_normalized = proj_A / level.max_aP
                    new_dim = self._create_polynomial_expansion(proj_A_normalized, level.d_proj)
        
        return distances
    
    def evaluate_single(self, spectral_value: np.ndarray) -> np.ndarray:
        """
        Avalia um único valor espectral (conveniência)
        
        Args:
            spectral_value: array (n_bands,) com valores das bandas
            
        Returns:
            array (num_levels,) com distâncias acumuladas
        """
        distances = self.evaluate(spectral_value)
        return distances[0]
    
    def evaluate_image(self, image_multispectral: np.ndarray) -> np.ndarray:
        """
        Avalia imagem inteira com cache de valores espectrais
        
        Args:
            image_multispectral: array (H, W, n_bands) com valores espectrais
            
        Returns:
            array (H, W) com distância do último nível (acumulada)
        """
        H, W = image_multispectral.shape[:2]
        
        # Valida número de bandas
        if image_multispectral.shape[2] != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, got {image_multispectral.shape[2]}")
        
        # Cache de combinações espectrais já processadas
        spectral_cache = {}
        result = np.zeros((H, W), dtype=np.float32)
        
        logger.info(f"Processing image of shape {image_multispectral.shape}")
        
        for i in range(H):
            if i % 100 == 0:
                logger.info(f"Processing row {i}/{H} ({100*i/H:.1f}%)")
                
            for j in range(W):
                pixel = tuple(image_multispectral[i, j])
                
                if pixel not in spectral_cache:
                    distances = self.evaluate(np.array(pixel, dtype=np.float32))
                    # Pega última distância (acumulada de todos os níveis)
                    spectral_cache[pixel] = distances[0, -1]
                
                result[i, j] = spectral_cache[pixel]
        
        logger.info(f"Processed {len(spectral_cache)} unique spectral signatures")
        return result
    
    def _log_levels_info(self) -> None:
        """Log informações sobre os níveis criados"""
        logger.info("=" * 60)
        logger.info("TOPOLOGICAL SPACE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Number of bands: {self.n_bands}")
        logger.info(f"Center: {self.center}")
        logger.info(f"Number of levels: {len(self.levels)}")
        
        for idx, level in enumerate(self.levels):
            logger.info(f"\n--- Level {idx + 1} ---")
            if level.A_basis is not None:
                logger.info(f"  A_basis shape: {level.A_basis.shape}")
            logger.info(f"  max_aP: {level.max_aP:.6f}")
            logger.info(f"  sigma_inv: {level.sigma_inv:.6e}")
            logger.info(f"  d_proj: {level.d_proj}")
            if level.dms is not None:
                logger.info(f"  dms shape: {level.dms.shape}")
                logger.info(f"  dms values: {level.dms}")
            if level.ind_use is not None:
                logger.info(f"  ind_use: {level.ind_use} (length: {len(level.ind_use)})")
        
        logger.info("=" * 60)