local KLD_Gaussian_Criterion, parent = torch.class("nn.KLD_Gaussian_Criterion", "nn.Criterion")
    
function KLD_Gaussian_Criterion:updateOutput(input)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	local mean = input[1]
    local logVar = input[2]

    local loss = logVar:clone()
    loss:exp():mul(-1) 					--   -sigma^2
    loss:add(-1, torch.pow(mean, 2) )   --   -mu^2 -sigma^2
    loss:add(1):add(logVar) 			--   1 + log(sigma^2) - mu^2 - sigma^2
        
    self.output = -0.5 * loss:sum()
    return self.output
end
    
function KLD_Gaussian_Criterion:updateGradInput(input)
    self.gradInput = {}
    local mean = input[1]
    local logVar = input[2]

    self.gradInput[1] = mean:clone()
    self.gradInput[2] = logVar:exp():mul(-1):add(1):mul(-0.5)

    return self.gradInput
end
