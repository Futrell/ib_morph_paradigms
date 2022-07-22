import torch

def ib_blahut_arimoto(W: int, gamma, lnp_M, lnp_U_given_M, num_steps=10, init_temperature=1, debug=False):
  # add dummy dimensions to make things easy
  U_dim, M_dim, W_dim = -3, -2, -1
  lnp_M = lnp_M[None, :, None] # shape 1M1
  lnp_U_given_M = lnp_U_given_M[:, :, None] # shape UM1

  # q(w|m) is an M x W matrix
  lnq = ((1/init_temperature)*torch.randn(1, lnp_M.shape[M_dim], W)).log_softmax(W_dim) # shape 1MW

  for i in range(num_steps):
    # start by getting q(m,w) = p(m) q(w|m)
    lnq_joint = lnp_M + lnq
    
    # get q0(w) = \sum_m q(m, w)
    lnq0 = lnq_joint.logsumexp(M_dim, keepdim=True) # shape 11W

    # to get the KL divergence,
    # first need p(m | w) =  q(m, w) / q0(w)
    lnq_inv = lnq_joint - lnq0 # shape 1MW

    # now need q(u|w) = \sum_m p(m | w) p(u | m)
    lnquw = (lnq_inv + lnp_U_given_M).logsumexp(M_dim, keepdim=True) # shape U1W

    # now need \sum_u p(u|m) ln q(u|w); use torch.xlogy for 0*log0 case
    d = -(lnp_U_given_M.exp() * lnquw).sum(U_dim, keepdim=True) # shape 1MW

    if debug:
      breakpoint()

    # finally get the encoder
    lnq = (lnq0 - gamma*d).log_softmax(W_dim) # shape 1MW

  return lnq.squeeze(U_dim) # remove dummy U dimension

cost = torch.Tensor([
  [0, 1, 2],
  [1, 0, 1],
  [2, 1, 0],                      
])

lnp_U_given_M = torch.log_softmax(-cost, -1)
lnp_U_given_M.exp()

lnp_M = torch.log(torch.ones(3) / 3)
lnp_M.exp()

# 3-word encoder for meanings: should produce p(w | m) = 1 or 0.
ib_blahut_arimoto(3, 1000, lnp_M, lnp_U_given_M).exp()

