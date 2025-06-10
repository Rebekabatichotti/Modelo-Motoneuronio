# Gerador de Gráfico ISI CV vs Firing Rate

Este script Python gera um gráfico de dispersão com linhas de tendência para analisar a relação entre ISI CV (Coefficient of Variation) e Firing Rate (Taxa de Disparo) em três condições diferentes.

## Arquivos Necessários

Certifique-se de que os seguintes arquivos CSV estejam no mesmo diretório:
- `fr_cv_normal.csv`
- `fr_cv_low_affected.csv`
- `fr_cv_severe.csv`

## Dependências

```bash
pip install pandas matplotlib numpy scipy
```

## Como Usar

1. **Execute o script:**
```bash
python generate_cv_fr_graph.py
```

2. **O script irá:**
   - Carregar os três arquivos CSV
   - Testar 5 modelos matemáticos diferentes (Linear, Exponencial, Logarítmico, Potência, Polinomial)
   - Selecionar o modelo com melhor ajuste (maior R²) para cada condição
   - Gerar um gráfico com os dados e linhas de tendência
   - Salvar o gráfico como `cv_fr_scatter_best_fit.png`
   - Imprimir as equações matemáticas no terminal

## Modelos Testados

1. **Linear:** `y = ax + b`
2. **Exponencial:** `y = a × e^(bx) + c`
3. **Logarítmico:** `y = a × ln(x) + b`
4. **Potência:** `y = a × x^b`
5. **Polinomial:** `y = ax² + bx + c`

## Saída Esperada

O script gera:
- **Gráfico:** `cv_fr_scatter_best_fit.png` (alta resolução, 300 DPI)
- **Terminal:** Estatísticas detalhadas e equações para cada condição

### Exemplo de Saída no Terminal:
```
Normal:
  Best fit: Logarithmic
  R²: 0.9817
  Equation: Firing_Rate = -6.174 * ln(ISI_CV) + 0.144

Low Affected:
  Best fit: Logarithmic
  R²: 0.9816
  Equation: Firing_Rate = -6.040 * ln(ISI_CV) - 0.211

Severe:
  Best fit: Logarithmic
  R²: 0.9797
  Equation: Firing_Rate = -5.830 * ln(ISI_CV) - 0.644
```

## Interpretação dos Resultados

- **R²:** Indica a qualidade do ajuste (0 a 1, sendo 1 o ajuste perfeito)
- **Eixo X:** ISI CV (Coefficient of Variation)
- **Eixo Y:** Firing Rate (Hz)
- **Linhas tracejadas:** Curvas de melhor ajuste para cada condição

## Personalização

Para modificar cores, tamanhos ou outros aspectos visuais, edite as variáveis no início da função `main()`:

```python
colors = {'Normal': '#1f77b4', 'Low Affected': '#ff7f0e', 'Severe': '#2ca02c'}
plt.figure(figsize=(12, 8))  # Tamanho da figura
```

## Estrutura dos Dados CSV

Os arquivos CSV devem ter as seguintes colunas:
- `firing_rate`: Taxa de disparo em Hz
- `ISI_CV`: Coeficiente de variação do intervalo inter-spike
- `neuron_index`: Índice do neurônio (opcional para análise)

## Troubleshooting

**Erro "ModuleNotFoundError":**
```bash
pip install pandas matplotlib numpy scipy
```

**Erro "FileNotFoundError":**
- Verifique se os arquivos CSV estão no mesmo diretório do script
- Confirme os nomes dos arquivos: `fr_cv_normal.csv`, `fr_cv_low_affected.csv`, `fr_cv_severe.csv`

**Gráfico não aparece:**
- O arquivo PNG é sempre salvo, mesmo se a janela não abrir
- Verifique o arquivo `cv_fr_scatter_best_fit.png` no diretório
