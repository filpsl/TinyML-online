// BIBLIOTECAS NECESSÁRIAS
#include <Arduino.h>
#include <math.h>

// Garante que M_PI está definido, embora a biblioteca do Arduino geralmente já o defina.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =================================================================
// PARÂMETROS GLOBAIS
// =================================================================

// PARÂMETROS DA REDE NEURAL
#define N_ENTRADA 1
#define N_OCULTAS 2
#define N_SAIDA 1
#define N_PONTOS_DE_DADOS 70

// PARÂMETROS DO ALGORITMO PSO (Particle Swarm Optimization)
#define S 7 // (S) Número de partículas no enxame
#define N 7   // (N) Número de dimensões (pesos + biases). (1*2 + 2) + (2*1 + 1) = 7
#define ITERACOES_MAX 500 // Número máximo de iterações

// =================================================================
// DADOS DE TREINO E VARIÁVEIS GLOBAIS
// =================================================================

// Dados de treino globais
float X_TREINO[N_PONTOS_DE_DADOS];
float Y_TREINO[N_PONTOS_DE_DADOS];

// Variáveis do PSO - Alocadas estaticamente para evitar problemas de memória no Arduino
float x[S][N];      // Posição atual das partículas
float y[S][N];      // Melhor posição local de cada partícula (pbest)
float v[S][N];      // Velocidade de cada partícula
float erros[S];     // Erro de cada partícula na sua melhor posição local

// Guarda o melhor resultado encontrado
float melhor_particula[N];

// =================================================================
// FUNÇÕES AUXILIARES
// =================================================================

// Função para gerar dados de uma senoide com ruído
void gerar_dados_seno() {
    for (uint8_t i = 0; i < N_PONTOS_DE_DADOS; i++) {
        X_TREINO[i] = -M_PI + (2 * M_PI * i) / (N_PONTOS_DE_DADOS - 1);
        // Gera um número aleatório entre 0.0 e 1.0 e adiciona ruído
        float ruido = (random(0, 1001) / 1000.0) * 0.3 - 0.15;
        Y_TREINO[i] = sin(X_TREINO[i]) + ruido;
    }
}

// Função que retorna um número aleatório de ponto flutuante em um intervalo
float rand_float(float min, float max) {
    return min + ((float)random(0, 10001) / 10000.0) * (max - min);
}

// =================================================================
// FUNÇÕES DA REDE NEURAL E FITNESS
// =================================================================

// Calcula a saída da rede neural para uma dada entrada e um conjunto de pesos/biases (partícula)
float rede_neural(const float particula[], float x_input) {
    // Extrai os pesos e biases do vetor da partícula usando ponteiros
    const float *w1 = &particula[0];
    const float *b1 = &particula[N_ENTRADA * N_OCULTAS];
    const float *w2 = &particula[N_ENTRADA * N_OCULTAS + N_OCULTAS];
    const float *b2 = &particula[N_ENTRADA * N_OCULTAS + N_OCULTAS + N_OCULTAS * N_SAIDA];

    float a1[N_OCULTAS]; // Ativações da camada oculta
    // Calcula a saída da camada oculta
    for (uint8_t i = 0; i < N_OCULTAS; i++) {
        a1[i] = tanh(x_input * w1[i] + b1[i]);
    }

    float z2 = 0.0f; // Saída final
    // Calcula a saída da camada final
    for (uint8_t i = 0; i < N_OCULTAS; i++) {
        z2 += a1[i] * w2[i];
    }
    z2 += b2[0];

    return z2;
}

// Função de avaliação (fitness): Calcula o Erro Quadrático Médio (MSE)
float funcao_fitness_mse(const float particula[]) {
    float erro_medio_quadratico = 0.0f;
    for (uint8_t i = 0; i < N_PONTOS_DE_DADOS; i++) {
        float y_calculado = rede_neural(particula, X_TREINO[i]);
        erro_medio_quadratico += pow(y_calculado - Y_TREINO[i], 2);
    }
    return erro_medio_quadratico / N_PONTOS_DE_DADOS;
}


// =================================================================
// ALGORITMO PSO (Particle Swarm Optimization)
// =================================================================

void PSO(float ys_posicao[]) {
    // Variáveis do PSO
    float c1 = 2.05;
    float c2 = 2.05;
    float w = 1.0;
    float w_final = 0.001;
    float w_passo = (w_final - w) / (ITERACOES_MAX);
    float vel_max = 5.0;
    float vel_final = 0.01;
    float vel_passo = (vel_final - vel_max) / (ITERACOES_MAX);
    uint8_t x_max = 4;
    int8_t x_min = -4;

    float ys_pos[N]; // Melhor posição global (gbest)
    float ys_erro;

    float ERRO_GLOBAL = 1;

    while (ERRO_GLOBAL > 0.03){
      // 1. Iniciando as partículas
      for (uint8_t i = 0; i < S; i++) {
          for (uint8_t j = 0; j < N; j++) {
              x[i][j] = rand_float(x_min, x_max);
              y[i][j] = x[i][j]; // A melhor posição inicial é a própria posição
              v[i][j] = 0;       // Velocidade inicial zero
          }
          // Avaliando o erro inicial de cada partícula
          erros[i] = funcao_fitness_mse(x[i]);
      }

      // 2. Definindo a melhor partícula global inicial
      uint8_t ys_indice = 0;
      ys_erro = erros[0];
      for (uint8_t i = 1; i < S; i++) {
          if (erros[i] < ys_erro) {
              ys_indice = i;
              ys_erro = erros[i];
          }
      }
      // Atribuindo a ys_pos a posição da melhor partícula global (gbest)
      for (uint8_t j = 0; j < N; j++) {
          ys_pos[j] = x[ys_indice][j];
      }

      // 3. Começando o loop principal de iterações
      for (uint16_t iteracao = 0; iteracao < ITERACOES_MAX; iteracao++) {
          // Itera por todas as partículas
          for (uint8_t i = 0; i < S; i++) {
              // Itera por todas as dimensões de cada partícula
              for (uint8_t j = 0; j < N; j++) {
                  float r1 = rand_float(0, 1);
                  float r2 = rand_float(0, 1);

                  // Atualizando a velocidade da partícula
                  v[i][j] = w * v[i][j] + c1 * r1 * (y[i][j] - x[i][j]) + c2 * r2 * (ys_pos[j] - x[i][j]);

                  // Limitando a velocidade máxima
                  if (fabs(v[i][j]) > vel_max) {
                      v[i][j] = (v[i][j] > 0) ? vel_max : -vel_max;
                  }

                  // Finalmente, atualizar a posição da partícula
                  x[i][j] = x[i][j] + v[i][j];

                  if (x[i][j] > x_max){
		  		          x[i][j] = x_max - 0.2*(rand_float(0,1));
		  		        }
		  		        else if (x[i][j] < x_min){
		  		          x[i][j] = x_min + 0.2*(rand_float(0,1));
		  		        }
              }
          }

          // 4. Avalia as partículas em suas novas posições
          for (uint8_t p = 0; p < S; p++) {
              float erro_da_particula = funcao_fitness_mse(x[p]);

              // Se o erro atual (pbest) for melhor, atualiza
              if (erro_da_particula < erros[p]) {
                  erros[p] = erro_da_particula;
                  for (uint8_t j = 0; j < N; j++) {
                      y[p][j] = x[p][j];
                  }
              }

              // Se o erro atual for melhor que o erro global (gbest), atualiza
              if (erros[p] < ys_erro) {
                  ys_erro = erros[p];
                  for (uint8_t j = 0; j < N; j++) {
                      ys_pos[j] = x[p][j];
                  }
              }
          }

          // Atualiza os parâmetros w e vel_max
          w += w_passo;
          vel_max += vel_passo;

          // Imprime o menor erro a cada 50 iterações
          if ((iteracao + 1) % 50 == 0) {
              Serial.print("Iteracao: ");
              Serial.print(iteracao + 1);
              Serial.print(" - Menor Erro: ");
              Serial.println(ys_erro, 6); // Imprime com 6 casas decimais
          }
      }

      // Guarda a melhor posição global encontrada
      for (uint8_t j = 0; j < N; j++) {
          ys_posicao[j] = ys_pos[j];
      }
    
    ERRO_GLOBAL = ys_erro;
    Serial.println("LOOP CONCLUÍDO!!");
    if (ERRO_GLOBAL > 0.03){
      Serial.print("ERRO GLOBAL: ");
      Serial.println(ERRO_GLOBAL);
      Serial.println("COMEÇANDO NOVAMENTE!");
    }
  }
} 

// =  ================================================================
// F  UNÇÕES PRINCIPAIS DO ARDUINO: setup() e loop()
// =================================================================

void setup() {
    // Inicia a comunicação serial para podermos ver os resultados no Serial Monitor
    Serial.begin(115200);
    while (!Serial) {
        ; // Espera a porta serial conectar. Necessário para placas como Leonardo.
    }
    delay(1000); // Pequena pausa para garantir que o monitor serial esteja pronto

    // Semente para o gerador de números aleatórios usando uma porta analógica flutuante
    randomSeed(analogRead(A0));
    
    Serial.println("Iniciando o treinamento da Rede Neural com PSO...");
    
    // Preparar os dados de treino
    gerar_dados_seno();
    Serial.println("Dados de treino gerados.");

    // Chama o algoritmo de otimização
    PSO(melhor_particula);

    Serial.println("\n--- Otimizacao Concluida ---");
    Serial.print("Menor Erro Final: ");
    Serial.println(funcao_fitness_mse(melhor_particula), 6);
    
    Serial.println("Melhor Particula (Pesos e Vieses):");
    for (uint8_t j = 0; j < N; j++) {
        Serial.print(melhor_particula[j], 4);
        Serial.print(" ");
    }
    Serial.println("\n---------------------------------");
}

void loop() {
    // O loop fica vazio, pois o treinamento é executado apenas uma vez no setup.
}