def moe():
    import torch
    from torch import nn
    import torch.nn.functional as F
    # sequences: ['What is 1 + 1 ?','Mixture of experts are typically FFNNs'] 
    # tokenized: [['What', 'is', '1', '+', '1', '?'],['Mixture', 'of', 'experts', 'are', 'typically', 'FFNNs']]
    hidden_states=torch.normal(2, 3, size=(2, 6,5)) # (batch_size, seq_len, emb_dim), each token being represented as a 5 dimensional vector
    print("Hidden_states:\n",hidden_states) 
    num_experts=4 # each expert will be a FeedForward Neural Network (FFNN)
    hidden_size=5 # hidden dimension
    intermediate_size=20 # the intermediate dimension in the FFNN

    # define an expert as a FFNN (Feedforward neural network)
    class Expert(nn.Module):
        def __init__(self, intermediate_size,hidden_size):
            super().__init__()
            self.ffn_dim = intermediate_size
            self.hidden_dim = hidden_size

            self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
            self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
            self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

            self.act_fn = nn.SiLU()

        def forward(self, hidden_states):
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
            current_hidden_states = self.w2(current_hidden_states)
            return current_hidden_states
        
    experts = nn.ModuleList([Expert(intermediate_size,hidden_size=5) for _ in range(num_experts)]) # define all the experts as a list of torch modules

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    top_k=2 # top 2 experts chosen by every token 

    jitter_noise= 0.01 # noise added to the hidden states to prevent the same experts from always being picked

    gate = nn.Linear(hidden_dim, num_experts, bias=False) # router network also called gate network, it Determines which tokens are sent to which experts.

    # add noise to the hidden states to prevent the same experts from always being picked
    if jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - jitter_noise, 1.0 + jitter_noise)

    hidden_states = hidden_states.view(-1, hidden_dim)


    router_logits = gate(hidden_states) # router_logits: (batch * sequence_length, n_experts)
    print("Router logits:\n",router_logits)

    # probabilities for every expert
    # for every token, a vector with dimension "num_experts" consists of probabilities
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) 
    print("Routing weights:\n",routing_weights)

    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # indices of the selected experts for every token 
    print("Selected experts indices:\n",selected_experts) # (batch * sequence_length, top_k)

    print("Selected routing weights:\n",routing_weights) # (batch * sequence_length, top_k)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # normalize the weights

    routing_weights = routing_weights.to(hidden_states.dtype) # cast back to the input dtype for data consistency

    # initialized final hidden states with zeros
    # this will be the output of a MoE layer
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)


    # Loop over all available experts and perform the computation on each expert
    for expert_idx in range(num_experts):
        expert_layer = experts[expert_idx]
        print("#"*50)
        print("Expert index:\n",expert_idx)

        idx, top_x = torch.where(expert_mask[expert_idx])  # idx and top_x will be of same size, both will be a tensor

        print("Current selected routing weight indices:\n",idx)
        print("Current selected Token indices:\n",top_x)


        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim) # select the current hidden states for the current tokens
        print(f"Current selected hidden states for the current selected token indices {top_x}:\n",current_state)

        # Make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        print(f"Current routing weights for the current selected token indices {top_x}:\n",routing_weights[top_x, idx, None])
        print(f"Current_hidden_states (output of the current expert layer {expert_idx}):\n",current_hidden_states) # output of the expert layer

        # index_add_ function does the following
        # if dim == 0, top_x[i] == j,
        # then the ith row of current_hidden_states is added from the jth row of final_hidden_states
        # example, i=0 and top_x[0]=4
        # 4 is the current token index
        # current_hidden_states[0] vector will be added to final_hidden_states[4] vector
        # let's say a token with index 4 has selected expert 0 and expert 1,
        # then the function index_add_ will concatenate (add) the output of the expert 0 and expert 1
        # a python equivalent of index_add_ for dim=0
        '''def index_add(final_hidden_states, current_hidden_states, source):
            for i, idx in enumerate(indices):
                final_hidden_states[idx] += current_hidden_states[i]

        # Initialize a list with 5 zeros
        final_hidden_states = [0, 0, 0, 0, 0] # x

        # Indices where values should be added
        indices = [1, 3,1]

        # Values to add at the specified indices
        current_hidden_states = [10, 20,10]  # Add 10 to x[1], 20 to x[3]. 10 to x[1]

        # Perform the operation
        index_add(final_hidden_states, indices, current_hidden_states)

        # Print the updated list
        print(final_hidden_states)'''

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    print("*"*50)
    print("final_hidden_states:\n",final_hidden_states)

if __name__ == "__main__":
    moe()
