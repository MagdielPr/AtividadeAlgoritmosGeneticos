import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple
from scipy import stats  

class ProblemaTroco:
    def __init__(self, valor_alvo: float, moedas: List[float]):
        self.valor_alvo = int(round(valor_alvo * 100))
        self.moedas = [int(round(m * 100)) for m in moedas]
        self.num_moedas = len(moedas)
        self.solucao_otima = self._calcular_solucao_otima()
        
    def _calcular_solucao_otima(self) -> List[int]:
        valor_restante = self.valor_alvo
        solucao = [0] * self.num_moedas
        indices_ordenados = sorted(range(self.num_moedas), 
                                 key=lambda i: self.moedas[i], reverse=True)
        
        for i in indices_ordenados:
            if valor_restante >= self.moedas[i]:
                quantidade = valor_restante // self.moedas[i]
                solucao[i] = quantidade
                valor_restante -= quantidade * self.moedas[i]
        return solucao
        
    def fitness(self, cromossomo: List[int]) -> float:
        valor_total = sum(cromossomo[i] * self.moedas[i] for i in range(self.num_moedas))
        num_moedas_total = sum(cromossomo)
        diferenca = abs(valor_total - self.valor_alvo)
        
        if diferenca > 0:
            return 50.0 / (1 + diferenca * 0.3 + num_moedas_total * 0.05)
        return 100.0 / (1 + num_moedas_total)
    
    def gerar_cromossomo_aleatorio(self) -> List[int]:
        cromossomo = []
        for moeda_valor in self.moedas:
            if moeda_valor > 0:
                limite = min(100, self.valor_alvo // moeda_valor + 10)
                cromossomo.append(random.randint(0, limite))
            else:
                cromossomo.append(random.randint(0, 10))
        return cromossomo
    
    def reparar_cromossomo(self, cromossomo: List[int]) -> List[int]:
        cromossomo_reparado = cromossomo.copy()
        valor_atual = sum(cromossomo_reparado[i] * self.moedas[i] for i in range(self.num_moedas))
        
        indices_ordenados = sorted(range(self.num_moedas), 
                                 key=lambda i: self.moedas[i], reverse=True)
        
        for i in indices_ordenados:
            while valor_atual > self.valor_alvo and cromossomo_reparado[i] > 0:
                if valor_atual - self.moedas[i] >= self.valor_alvo:
                    cromossomo_reparado[i] -= 1
                    valor_atual -= self.moedas[i]
                else:
                    break
        
        valor_restante = self.valor_alvo - valor_atual
        for i in indices_ordenados:
            while valor_restante >= self.moedas[i]:
                cromossomo_reparado[i] += 1
                valor_restante -= self.moedas[i]
                
        return cromossomo_reparado

def calcular_diversidade(populacao: List[List[int]]) -> float:
    """Calcula diversidade genética como distância hamming média"""
    if len(populacao) < 2:
        return 0.0
    
    diversidade_total = 0
    comparacoes = 0
    
    for i in range(len(populacao)):
        for j in range(i+1, len(populacao)):
            diferencias = sum(1 for k in range(len(populacao[i])) 
                            if populacao[i][k] != populacao[j][k])
            diversidade_total += diferencias
            comparacoes += 1
    
    return diversidade_total / comparacoes if comparacoes > 0 else 0.0

class AlgoritmoGeneticoTradicional:
    def __init__(self, problema: ProblemaTroco, tamanho_populacao: int = 150, 
                 taxa_cruzamento: float = 0.85, taxa_mutacao: float = 0.005):
        self.problema = problema
        self.tamanho_populacao = tamanho_populacao
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.populacao = []
        self.historico_fitness = []
        self.historico_moedas = []
        self.historico_diversidade = []  # Histórico de diversidade genética
        self.geracao_convergencia = 0  # Geração onde convergiu
        
    def inicializar_populacao(self):
        self.populacao = []
        
        # Soluções gulosas variadas (30%)
        for _ in range(max(1, int(self.tamanho_populacao * 0.3))):
            cromossomo = [0] * self.problema.num_moedas
            valor_restante = self.problema.valor_alvo
            moedas_shuffled = list(range(self.problema.num_moedas))
            random.shuffle(moedas_shuffled)
            
            for i in moedas_shuffled:
                if valor_restante > 0 and self.problema.moedas[i] > 0:
                    max_q = valor_restante // self.problema.moedas[i]
                    cromossomo[i] = random.randint(0, min(max_q, 20))
                    valor_restante -= cromossomo[i] * self.problema.moedas[i]
            self.populacao.append(self.problema.reparar_cromossomo(cromossomo))
        
        # Restante aleatório
        while len(self.populacao) < self.tamanho_populacao:
            cromossomo = self.problema.gerar_cromossomo_aleatorio()
            self.populacao.append(self.problema.reparar_cromossomo(cromossomo))
    
    def selecao_torneio(self, tamanho_torneio: int = 5) -> List[int]:
        competidores = random.sample(range(self.tamanho_populacao), tamanho_torneio)
        melhor_idx = max(competidores, key=lambda i: self.problema.fitness(self.populacao[i]))
        return self.populacao[melhor_idx].copy()
    
    def cruzamento_pontos(self, pai1: List[int], pai2: List[int]) -> Tuple[List[int], List[int]]:
        tamanho = len(pai1)
        ponto1 = random.randint(1, tamanho-2)
        ponto2 = random.randint(ponto1+1, tamanho-1)
        filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
        filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
        return (
            self.problema.reparar_cromossomo(filho1),
            self.problema.reparar_cromossomo(filho2)
        )
    
    def mutacao_estrategica(self, cromossomo: List[int]) -> List[int]:
        cromossomo_mutado = cromossomo.copy()
        for i in range(len(cromossomo)):
            if random.random() < self.taxa_mutacao:
                alteracao = random.randint(-2, 2)
                cromossomo_mutado[i] = max(0, cromossomo_mutado[i] + alteracao)
        return self.problema.reparar_cromossomo(cromossomo_mutado)
    
    def evoluir(self, num_geracoes: int) -> Tuple[List[int], List[float], List[int]]:
        self.inicializar_populacao()
        self.historico_fitness = []
        self.historico_moedas = []
        self.historico_diversidade = []
        
        melhor_fitness_anterior = 0
        geracoes_sem_melhoria = 0
        limite_estagnacao = 300
        
        for geracao in range(num_geracoes):
            fitness_pop = [self.problema.fitness(ind) for ind in self.populacao]
            melhor_fitness = max(fitness_pop)
            melhor_idx = fitness_pop.index(melhor_fitness)
            num_moedas_melhor = sum(self.populacao[melhor_idx])
            self.historico_fitness.append(melhor_fitness)
            self.historico_moedas.append(num_moedas_melhor)
            
            # Calcular diversidade genética
            diversidade_atual = calcular_diversidade(self.populacao)
            self.historico_diversidade.append(diversidade_atual)
            
            if melhor_fitness > melhor_fitness_anterior:
                melhor_fitness_anterior = melhor_fitness
                geracoes_sem_melhoria = 0
                self.geracao_convergencia = geracao  # Registra geração de convergência
            else:
                geracoes_sem_melhoria += 1
            
            if geracoes_sem_melhoria >= limite_estagnacao:
                break
            
            nova_pop = []
            num_elite = max(2, int(self.tamanho_populacao * 0.1))
            indices_elite = sorted(range(len(fitness_pop)), 
                                 key=lambda i: fitness_pop[i], reverse=True)[:num_elite]
            nova_pop.extend([self.populacao[i].copy() for i in indices_elite])
            
            while len(nova_pop) < self.tamanho_populacao:
                pai1 = self.selecao_torneio()
                pai2 = self.selecao_torneio()
                
                if random.random() < self.taxa_cruzamento:
                    filho1, filho2 = self.cruzamento_pontos(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                
                nova_pop.append(self.mutacao_estrategica(filho1))
                if len(nova_pop) < self.tamanho_populacao:
                    nova_pop.append(self.mutacao_estrategica(filho2))
            
            self.populacao = nova_pop[:self.tamanho_populacao]
        
        fitness_final = [self.problema.fitness(ind) for ind in self.populacao]
        melhor_idx = fitness_final.index(max(fitness_final))
        return (
            self.populacao[melhor_idx],
            self.historico_fitness,
            self.historico_moedas
        )

class AlgoritmoGeneticoIlhas:
    """
    Algoritmo Genético com Estratégia de Ilhas
    
    CONFIGURAÇÃO DA ESTRATÉGIA:
    - Número de ilhas: 5 (configurável via num_ilhas)
    - Frequência de migração: a cada 10 gerações (configurável via freq_migracao)
    - Topologia de comunicação: circular (ilha i migra para ilha (i+1) % num_ilhas)
    - Taxa de migração: 10% dos melhores indivíduos de cada ilha (configurável via taxa_migracao)
    - Tamanho de cada ilha: 40 indivíduos (configurável via tamanho_ilha)
    
    DIVERSIFICAÇÃO POR ILHA:
    - Ilha 0: Estratégia conservadora (prioriza moedas de maior valor)
    - Ilha 1: Algoritmo guloso com variações (soluções semi-ótimas)
    - Ilhas 2-4: Populações completamente aleatórias (exploração)
    """
    def __init__(self, problema: ProblemaTroco, num_ilhas: int = 5, 
                 tamanho_ilha: int = 40, taxa_cruzamento: float = 0.85, 
                 taxa_mutacao: float = 0.005, freq_migracao: int = 10,
                 taxa_migracao: float = 0.1):
        self.problema = problema
        self.num_ilhas = num_ilhas
        self.tamanho_ilha = tamanho_ilha
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.freq_migracao = freq_migracao
        self.taxa_migracao = taxa_migracao
        self.ilhas = []
        self.historico_fitness = []
        self.historico_moedas = []
        self.historico_diversidade = []  # Histórico de diversidade genética
        self.geracao_convergencia = 0  # Geração onde convergiu
        
    def inicializar_ilhas(self):
        """
        Inicialização especializada por ilha:
        - Ilha 0: Estratégia conservadora (prioriza moedas grandes)
        - Ilha 1: Algoritmo guloso com variações
        - Ilhas 2-4: Populações completamente aleatórias
        """
        self.ilhas = []
        
        # Ilha 0: População conservadora - prioriza moedas grandes
        ilha_conservadora = []
        for _ in range(self.tamanho_ilha):
            crom = [0] * self.problema.num_moedas
            for i in range(min(3, self.problema.num_moedas)):
                max_quant = int(self.problema.valor_alvo / self.problema.moedas[i]) + 2
                crom[i] = random.randint(0, max_quant)
            ilha_conservadora.append(self.problema.reparar_cromossomo(crom))
        self.ilhas.append(ilha_conservadora)
        
        # Ilha 1: Soluções gulosas variadas - gera soluções com algoritmo guloso
        ilha_gulosa = []
        for _ in range(self.tamanho_ilha):
            crom = [0] * self.problema.num_moedas
            valor_restante = self.problema.valor_alvo
            moedas_shuffled = list(range(self.problema.num_moedas))
            random.shuffle(moedas_shuffled)
            
            for i in moedas_shuffled:
                if valor_restante > 0 and self.problema.moedas[i] > 0:
                    max_q = valor_restante // self.problema.moedas[i]
                    crom[i] = random.randint(0, min(max_q, 20))
                    valor_restante -= crom[i] * self.problema.moedas[i]
            ilha_gulosa.append(self.problema.reparar_cromossomo(crom))
        self.ilhas.append(ilha_gulosa)
        
        # Demais ilhas: Populações aleatórias para diversidade
        for _ in range(max(0, self.num_ilhas - 2)):
            ilha = []
            for _ in range(self.tamanho_ilha):
                crom = self.problema.gerar_cromossomo_aleatorio()
                ilha.append(self.problema.reparar_cromossomo(crom))
            self.ilhas.append(ilha)
    
    def evoluir_ilha(self, ilha_idx: int) -> List[List[int]]:
        ilha = self.ilhas[ilha_idx]
        fitness_ilha = [self.problema.fitness(ind) for ind in ilha]
        nova_ilha = []
        
        num_elite = max(1, int(self.tamanho_ilha * 0.1))
        elite_indices = sorted(range(len(fitness_ilha)), 
                            key=lambda i: fitness_ilha[i], reverse=True)[:num_elite]
        nova_ilha.extend([ilha[i].copy() for i in elite_indices])
        
        ag_trad = AlgoritmoGeneticoTradicional(self.problema, self.tamanho_ilha, taxa_mutacao=self.taxa_mutacao)
        ag_trad.populacao = ilha
        
        while len(nova_ilha) < self.tamanho_ilha:
            pai1 = ag_trad.selecao_torneio()
            pai2 = ag_trad.selecao_torneio()
            
            if random.random() < self.taxa_cruzamento:
                filho1, filho2 = ag_trad.cruzamento_pontos(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()
            
            nova_ilha.append(ag_trad.mutacao_estrategica(filho1))
            if len(nova_ilha) < self.tamanho_ilha:
                nova_ilha.append(ag_trad.mutacao_estrategica(filho2))
        
        return nova_ilha[:self.tamanho_ilha]
    
    def migrar(self):
        """
        PROCESSO DE MIGRAÇÃO CIRCULAR:
        1. Seleciona os 10% melhores indivíduos de cada ilha (taxa_migracao)
        2. Substitui os 10% piores indivíduos da ilha destino
        3. Topologia circular: ilha 0→1, ilha 1→2, ..., ilha (n-1)→0
        4. Objetivo: Compartilhar boas soluções mantendo diversidade
        """
        num_migrantes = max(1, int(self.tamanho_ilha * self.taxa_migracao))
        migrantes = []
        
        for ilha in self.ilhas:
            fitness_ilha = [self.problema.fitness(ind) for ind in ilha]
            indices_melhores = sorted(range(len(fitness_ilha)), 
                                    key=lambda i: fitness_ilha[i], reverse=True)[:num_migrantes]
            migrantes.append([ilha[i].copy() for i in indices_melhores])
        
        for i in range(self.num_ilhas):
            destino = (i + 1) % self.num_ilhas
            fitness_destino = [self.problema.fitness(ind) for ind in self.ilhas[destino]]
            indices_piores = sorted(range(len(fitness_destino)), 
                                 key=lambda i: fitness_destino[i])[:num_migrantes]
            
            num_disponiveis = min(len(indices_piores), len(migrantes[i]))
            for j in range(num_disponiveis):
                self.ilhas[destino][indices_piores[j]] = migrantes[i][j]
    
    def evoluir(self, num_geracoes: int) -> Tuple[List[int], List[float], List[int]]:
        self.inicializar_ilhas()
        self.historico_fitness = []
        self.historico_moedas = []
        self.historico_diversidade = []
        
        melhor_fitness_anterior = 0
        geracoes_sem_melhoria = 0
        limite_estagnacao = 300
        
        for geracao in range(num_geracoes):
            for i in range(self.num_ilhas):
                self.ilhas[i] = self.evoluir_ilha(i)
            
            if geracao > 0 and geracao % self.freq_migracao == 0:
                self.migrar()
            
            melhor_fitness_global = 0
            melhor_moedas_global = 0
            
            # Combinar todas as populações para calcular diversidade global
            todas_populacoes = []
            for ilha in self.ilhas:
                todas_populacoes.extend(ilha)
                
            # Calcular diversidade global
            diversidade_global = calcular_diversidade(todas_populacoes)
            self.historico_diversidade.append(diversidade_global)
            
            for ilha in self.ilhas:
                fitness_ilha = [self.problema.fitness(ind) for ind in ilha]
                melhor_fitness = max(fitness_ilha)
                if melhor_fitness > melhor_fitness_global:
                    melhor_fitness_global = melhor_fitness
                    melhor_idx = fitness_ilha.index(melhor_fitness)
                    melhor_moedas_global = sum(ilha[melhor_idx])
            
            self.historico_fitness.append(melhor_fitness_global)
            self.historico_moedas.append(melhor_moedas_global)
            
            if melhor_fitness_global > melhor_fitness_anterior:
                melhor_fitness_anterior = melhor_fitness_global
                geracoes_sem_melhoria = 0
                self.geracao_convergencia = geracao
            else:
                geracoes_sem_melhoria += 1
            
            if geracoes_sem_melhoria >= limite_estagnacao:
                break
        
        melhor_solucao = None
        melhor_fitness = 0
        for ilha in self.ilhas:
            fitness_ilha = [self.problema.fitness(ind) for ind in ilha]
            melhor_idx = fitness_ilha.index(max(fitness_ilha))
            if fitness_ilha[melhor_idx] > melhor_fitness:
                melhor_fitness = fitness_ilha[melhor_idx]
                melhor_solucao = ilha[melhor_idx]
        
        return melhor_solucao, self.historico_fitness, self.historico_moedas

def executar_experimentos_comparativos(valor_alvo: float = 8.37):
    moedas = [0.50, 0.25, 0.10, 0.05, 0.01]
    problema = ProblemaTroco(valor_alvo, moedas)
    sol_otima, num_moedas_otima = problema.solucao_otima, sum(problema.solucao_otima)
    
    num_execucoes = 5
    num_geracoes = 500
    
    # Armazenar instâncias para análise posterior
    ags_tradicionais = []
    ags_ilhas = []
    
    resultados = {
        'tradicional': {'solucoes': [], 'fitness': [], 'moedas': [], 'tempos': [], 'convergencia': [], 'diversidade_hist': []},
        'ilhas': {'solucoes': [], 'fitness': [], 'moedas': [], 'tempos': [], 'convergencia': [], 'diversidade_hist': []}
    }
    
    print(f"\n=== EXECUTANDO EXPERIMENTOS PARA R$ {valor_alvo:.2f} ===")
    
    for i in range(num_execucoes):
        print(f"\nExecução {i+1}/{num_execucoes}")
        
        # AG Tradicional
        inicio = time.time()
        ag_trad = AlgoritmoGeneticoTradicional(problema, tamanho_populacao=150, taxa_mutacao=0.005)
        sol_trad, hist_fit_trad, hist_moedas_trad = ag_trad.evoluir(num_geracoes)
        tempo_trad = time.time() - inicio
        
        ags_tradicionais.append(ag_trad)
        resultados['tradicional']['solucoes'].append(sol_trad)
        resultados['tradicional']['fitness'].append(hist_fit_trad)
        resultados['tradicional']['moedas'].append(sum(sol_trad))
        resultados['tradicional']['tempos'].append(tempo_trad)
        resultados['tradicional']['convergencia'].append(ag_trad.geracao_convergencia)
        resultados['tradicional']['diversidade_hist'].append(ag_trad.historico_diversidade)
        
        # AG Ilhas
        inicio = time.time()
        ag_ilhas = AlgoritmoGeneticoIlhas(problema)
        sol_ilhas, hist_fit_ilhas, hist_moedas_ilhas = ag_ilhas.evoluir(num_geracoes)
        tempo_ilhas = time.time() - inicio
        
        ags_ilhas.append(ag_ilhas)
        resultados['ilhas']['solucoes'].append(sol_ilhas)
        resultados['ilhas']['fitness'].append(hist_fit_ilhas)
        resultados['ilhas']['moedas'].append(sum(sol_ilhas))
        resultados['ilhas']['tempos'].append(tempo_ilhas)
        resultados['ilhas']['convergencia'].append(ag_ilhas.geracao_convergencia)
        resultados['ilhas']['diversidade_hist'].append(ag_ilhas.historico_diversidade)
    
    print("\nRESULTADOS COMPARATIVOS")
    print("="*60)
    
    trad_moedas = resultados['tradicional']['moedas']
    trad_tempos = resultados['tradicional']['tempos']
    trad_conv = resultados['tradicional']['convergencia']
    
    print("\nAG TRADICIONAL:")
    print(f"  Média moedas: {np.mean(trad_moedas):.1f} ± {np.std(trad_moedas):.2f}")
    print(f"  Melhor: {min(trad_moedas)} moedas")
    print(f"  Taxa sucesso: {sum(m == num_moedas_otima for m in trad_moedas)}/{num_execucoes}")
    print(f"  Tempo médio: {np.mean(trad_tempos):.2f}s ± {np.std(trad_tempos):.2f}s")
    print(f"  Geração média de convergência: {np.mean(trad_conv):.0f}")
    
    ilhas_moedas = resultados['ilhas']['moedas']
    ilhas_tempos = resultados['ilhas']['tempos']
    ilhas_conv = resultados['ilhas']['convergencia']
    
    print("\nAG ILHAS:")
    print(f"  Média moedas: {np.mean(ilhas_moedas):.1f} ± {np.std(ilhas_moedas):.2f}")
    print(f"  Melhor: {min(ilhas_moedas)} moedas")
    print(f"  Taxa sucesso: {sum(m == num_moedas_otima for m in ilhas_moedas)}/{num_execucoes}")
    print(f"  Tempo médio: {np.mean(ilhas_tempos):.2f}s ± {np.std(ilhas_tempos):.2f}s")
    print(f"  Geração média de convergência: {np.mean(ilhas_conv):.0f}")
    
    # Análise estatística mais robusta
    print("\nANÁLISE ESTATÍSTICA DETALHADA:")
    print("="*50)

    # Teste t para comparar desempenho
    t_stat, p_valor = stats.ttest_ind(trad_moedas, ilhas_moedas)
    print(f"Teste t (número de moedas): t={t_stat:.3f}, p={p_valor:.3f}")

    if p_valor < 0.05:
        melhor_algoritmo = "AG Ilhas" if np.mean(ilhas_moedas) < np.mean(trad_moedas) else "AG Tradicional"
        print(f"Diferença SIGNIFICATIVA - {melhor_algoritmo} é estatisticamente superior")
    else:
        print("Diferença NÃO significativa entre os algoritmos")

    print(f"\nCoeficiente de variação:")
    print(f"  AG Tradicional: {np.std(trad_moedas)/np.mean(trad_moedas)*100:.1f}%")
    print(f"  AG Ilhas: {np.std(ilhas_moedas)/np.mean(ilhas_moedas)*100:.1f}%")
    
    # Cálculo de diversidade para gráficos
    min_len_trad = min(len(hist) for hist in resultados['tradicional']['fitness'])
    min_len_ilhas = min(len(hist) for hist in resultados['ilhas']['fitness'])
    
    hist_trad_med = np.mean([hist[:min_len_trad] for hist in resultados['tradicional']['fitness']], axis=0)
    hist_ilhas_med = np.mean([hist[:min_len_ilhas] for hist in resultados['ilhas']['fitness']], axis=0)
    
    # Calcular média da diversidade para gráfico
    min_len_div = min(min(len(hist) for hist in resultados['tradicional']['diversidade_hist']),
                     min(len(hist) for hist in resultados['ilhas']['diversidade_hist']))
    
    div_trad_med = np.mean([hist[:min_len_div] for hist in resultados['tradicional']['diversidade_hist']], axis=0)
    div_ilhas_med = np.mean([hist[:min_len_div] for hist in resultados['ilhas']['diversidade_hist']], axis=0)
    
    # Gráficos
    plt.figure(figsize=(14, 10))
    
    # Gráfico de convergência média
    plt.subplot(2, 2, 1)
    plt.plot(hist_trad_med, 'b-', linewidth=2, label='AG Tradicional')
    plt.plot(hist_ilhas_med, 'r-', linewidth=2, label='AG Ilhas')
    plt.title('Evolução Média do Fitness')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Boxplot de distribuição de moedas
    plt.subplot(2, 2, 2)
    plt.boxplot([trad_moedas, ilhas_moedas], labels=['Tradicional', 'Ilhas'])
    plt.title('Distribuição do Número de Moedas')
    plt.ylabel('Número de Moedas')
    plt.axhline(y=num_moedas_otima, color='g', linestyle='--', label='Solução Ótima')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de evolução da diversidade
    plt.subplot(2, 2, 3)
    plt.plot(div_trad_med, 'b-', linewidth=2, label='AG Tradicional')
    plt.plot(div_ilhas_med, 'r-', linewidth=2, label='AG Ilhas')
    plt.title('Evolução da Diversidade Genética')
    plt.xlabel('Geração')
    plt.ylabel('Diversidade (Distância Hamming)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparação de desempenho
    plt.subplot(2, 2, 4)
    plt.bar(['Tradicional', 'Ilhas'], [np.mean(trad_tempos), np.mean(ilhas_tempos)], 
            color=['blue', 'red'], alpha=0.7)
    plt.title('Tempo Médio de Execução')
    plt.ylabel('Segundos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'resultados_{valor_alvo:.2f}.png')
    plt.show()
    
    return resultados

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    valor_alvo = 17.46  # Valor único configurável
    executar_experimentos_comparativos(valor_alvo)