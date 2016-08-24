immutable SeparableSum <: ProximableFunction
  N::Int
  fs::Tuple
  SeparableSum(fs...) =
    new(length(fs), fs)
end

function call(f::SeparableSum, xs...)
  vs = map(i -> f.fs[i](xs[i]), 1:f.N)
  return sum(vs)
end

function prox(f::SeparableSum, gamma::Float64, xs...)
  res = map(i -> prox(f.fs[i], gamma, xs[i]), (1:f.N...))
  return map(p -> p[1], res), sum(map(p -> p[2], res))
end

function Base.show(io::IO, f::SeparableSum)
  for i = 1:f.N-1
    println(io, f.fs[i])
    println(io, "---")
  end
  print(io, f.fs[f.N])
end
