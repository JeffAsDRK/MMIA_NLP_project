from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Función auxiliar para redimensionar el tensor de frecuencias para que tenga la misma forma 
    que el tensor objetivo 'x' con el propósito de hacer broadcasting del tensor de frecuencias 
    durante operaciones elemento por elemento.

    Args:
        freqs_cis (torch.Tensor): Tensor de frecuencias a redimensionar.
        x (torch.Tensor): Tensor objetivo para compatibilidad de broadcasting.

    Returns:
        torch.Tensor: Tensor de frecuencias redimensionado.

    Raises:
        AssertionError: Si el tensor de frecuencias no coincide con la forma esperada.
        AssertionError: Si el tensor objetivo 'x' no tiene el número esperado de dimensiones.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Crear forma para broadcasting: mantener dimensiones 1 y última, hacer 1 el resto
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica embeddings rotacionales (RoPE) a los tensores de entrada usando el tensor de frecuencias.

    Esta función aplica embeddings rotacionales a los tensores query y key dados. La rotación 
    de cada embedding de token es una función de la posición de ese token en la secuencia, 
    head_dim y theta. Los tensores de entrada se redimensionan como números complejos para 
    simplificar la implementación.

    Args:
        query (torch.Tensor): Tensor query para aplicar embeddings rotacionales.
                              Forma: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Tensor key para aplicar embeddings rotacionales.
                            Forma: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimensión de cada cabeza de atención.
        max_seq_len (int): Longitud máxima de secuencia soportada por el modelo.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tupla de tensores query y key modificados con embeddings rotacionales.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    
    # Redimensionar xq y xk para coincidir con la representación compleja
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # Esto separa cada vector query/key en sus índices impares y pares (asumiendo indexación desde 1).
    # query_real contiene q_1, q_3, q_5, ... y query_imag contiene q_2, q_4, q_6, ...

    # Primero, calcular los valores trigonométricos según la fórmula RoPE
    
    # Crear tensor de frecuencias: freqs = 1.0 / (theta^(2i/head_dim)) for i = 0, 1, ..., head_dim//2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    freqs = freqs.to(device)
    
    # Crear índices de posición para la secuencia
    t = torch.arange(seqlen, device=device, dtype=freqs.dtype)
    
    # Calcular producto externo para obtener freqs para todas las posiciones
    freqs = torch.outer(t, freqs)  # Forma: (seqlen, head_dim//2)
    
    # Calcular valores de coseno y seno
    freqs_cos = torch.cos(freqs)  # Forma: (seqlen, head_dim//2)
    freqs_sin = torch.sin(freqs)  # Forma: (seqlen, head_dim//2)
    
    # Luego, combinar estos valores trigonométricos con los tensores query_real, query_imag,
    # key_real, y key_imag.
    
    # Redimensionar freqs_cos y freqs_sin para coincidir con las dimensiones de query/key
    # Necesitamos redimensionar a (1, seqlen, 1, head_dim//2) para broadcasting
    freqs_cos = freqs_cos.view(1, seqlen, 1, head_dim // 2)
    freqs_sin = freqs_sin.view(1, seqlen, 1, head_dim // 2)
    
    # Aplicar embedding rotacional: rotar query y key usando rotación compleja
    # Para número complejo (a + bi), rotación por ángulo θ: (a + bi) * e^(iθ) = (a + bi) * (cos(θ) + i*sin(θ))
    # Parte real: a*cos(θ) - b*sin(θ)
    # Parte imaginaria: a*sin(θ) + b*cos(θ)
    
    query_out_real = query_real * freqs_cos - query_imag * freqs_sin
    query_out_imag = query_real * freqs_sin + query_imag * freqs_cos
    
    key_out_real = key_real * freqs_cos - key_imag * freqs_sin
    key_out_imag = key_real * freqs_sin + key_imag * freqs_cos
    
    # Recombinar partes reales e imaginarias de vuelta al formato de tensor original
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).flatten(-2)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).flatten(-2)
    # Devolver los embeddings de posición rotacionales para los tensores query y key
    return query_out, key_out
