require 'nn'

local SpatialCircularPadding, parent = torch.class('nn.SpatialCircularPadding', 'nn.Module')

function SpatialCircularPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l

   if self.pad_l < 0 or  self.pad_r < 0 or self.pad_t < 0 or self.pad_b < 0 then
      error('padding should be > 0')
   end

   self.active = true
end

function SpatialCircularPadding:updateOutput(input)

   if not self.active then
      self.output = input
      return self.output
   end

   if input:dim() ~= 3 then
      error('input must be 2 or 3-dimensional')
   end

   -- sizes
   local h = input:size(2) + self.pad_t + self.pad_b
   local w = input:size(3) + self.pad_l + self.pad_r

   if w < 1 or h < 1 then error('input is too small') end

   self.output = torch.zeros(input:size(1), h, w):cuda()

   -- crop input if necessary
   local c_input = input

   -- crop outout if necessary
   local c_output = self.output
   c_output = c_output:narrow(2, 1 + self.pad_t, c_output:size(2) - self.pad_t)
   c_output = c_output:narrow(2, 1, c_output:size(2) - self.pad_b)
   c_output = c_output:narrow(3, 1 + self.pad_l, c_output:size(3) - self.pad_l)
   c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_r)

   -- copy input to output
   c_output:copy(c_input)

   -----------------------------------------------------------------------
   -- It should be done like folowing, but it is not clear about corners,
   -- Filling them with 0 is bad idea, since NN will find the corners then
   -- So use a little weird version
   ------------------------------------------------------------------------

   -- local tb_slice = self.output:narrow(3, self.pad_l+1,  input:size(3))
   -- local lr_slice = self.output:narrow(2, self.pad_t+1,  input:size(2))

   -- tb_slice:narrow(2, 1, self.pad_t):copy(input:narrow(2, input:size(2) - self.pad_t + 1, self.pad_t))
   -- tb_slice:narrow(2, input:size(2) + self.pad_t + 1, self.pad_b):copy(input:narrow(2, 1, self.pad_b))

   -- lr_slice:narrow(3, 1, self.pad_l):copy(input:narrow(3, input:size(3) - self.pad_l + 1, self.pad_l))
   -- lr_slice:narrow(3, input:size(3) + self.pad_l + 1, self.pad_r):copy(input:narrow(3, 1, self.pad_r))

   -- zero out corners
   -- self.output:narrow(3, 1, self.pad_l):narrow(2, 1, self.pad_t):zero()
   -- self.output:narrow(3, 1, self.pad_l):narrow(2, input:size(2) + self.pad_t + 1, self.pad_b):zero()
   -- self.output:narrow(3, input:size(3) + self.pad_l + 1, self.pad_r):narrow(2, 1, self.pad_t):zero()
   -- self.output:narrow(3, input:size(3) + self.pad_l + 1, self.pad_r):narrow(2, input:size(2) + self.pad_t + 1, self.pad_b):zero()

   -----------------------------------------------------------------------
   -- About right, but fills corners with something ..
   -----------------------------------------------------------------------

   self.output:narrow(2,1,self.pad_t):copy(self.output:narrow(2,input:size(2) + 1,self.pad_t))
   self.output:narrow(2,input:size(2) + self.pad_t + 1,self.pad_b):copy(self.output:narrow(2,self.pad_t + 1,self.pad_b))

   self.output:narrow(3,1,self.pad_l):copy(self.output:narrow(3,input:size(3) + 1,self.pad_l))
   self.output:narrow(3,input:size(3) + self.pad_l + 1,self.pad_r):copy(self.output:narrow(3,self.pad_l+1,self.pad_r))



   return self.output
end

function SpatialCircularPadding:updateGradInput(input, gradOutput)
   if not self.active then
      self.gradInput = gradOutput:clone()
      return self.gradInput
   end

   if input:dim() ~= 3 then
      error('input must be 2 or 3-dimensional')
   end

   -- Do it inplace to save memory
   self.gradInput = nil
   local cg_output = gradOutput

   cg_output = cg_output:narrow(2, 1 + self.pad_t, cg_output:size(2) - self.pad_t)
   cg_output = cg_output:narrow(2, 1, cg_output:size(2) - self.pad_b)
   cg_output = cg_output:narrow(3, 1 + self.pad_l, cg_output:size(3) - self.pad_l)
   cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_r)


   -- Border gradient
   local tb_slice = gradOutput:narrow(3, self.pad_l+1,  input:size(3))
   local lr_slice = gradOutput:narrow(2, self.pad_t+1,  input:size(2))

   cg_output:narrow(2, input:size(2) - self.pad_t + 1, self.pad_t):add(tb_slice:narrow(2, 1, self.pad_t))
   cg_output:narrow(2, 1, self.pad_b):add(tb_slice:narrow(2, input:size(2) + self.pad_t + 1, self.pad_b))

   cg_output:narrow(3, input:size(3) - self.pad_l + 1, self.pad_l):add(lr_slice:narrow(3, 1, self.pad_l))
   cg_output:narrow(3, 1, self.pad_r):add(lr_slice:narrow(3, input:size(3) + self.pad_l + 1, self.pad_r))


   self.gradInput = cg_output

   return self.gradInput
end


function SpatialCircularPadding:__tostring__()
  return torch.type(self) ..
      string.format('(l=%d,r=%d,t=%d,b=%d)', self.pad_l, self.pad_r,
                    self.pad_t, self.pad_b)
end
