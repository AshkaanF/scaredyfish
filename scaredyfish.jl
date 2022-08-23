"""
Simulates visual motion-based social networks among individual agents in continuous space.

- AKF, 2022

"""

using Distributions,
LinearAlgebra,
Distances,
NearestNeighbors,
StaticArrays,
SparseArrays,
IterTools,
DelimitedFiles


"""
[1] Model initialization __________________________________________________________________________________________________________________________________________

"""


"""
A speedier findall variant.

Parameters
----------
function : e.g., y->y==0.
Array : object you want to findall in.

Returns
-------
{Int} indices that meet criterion

"""
function findallF(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        @inbounds if f(a[i])
            b[j] = i
            j += 1
        end
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end


"""
Return N smallest entries in an array

Parameters
----------
array : array to sort
N : number desired

Returns
-------
Value

"""
function smallestN(a, n::Int)
  sort(a; alg = Base.Sort.PartialQuickSort(n))[n]
end


"""
Define our agent composite data object.

Fields
----------
x : current x coordinate of agent
y : current y coordinate of agent
h : heading of agent, in radians [-π, π]
s : state of agent. 0 for feeding, 1 for fleeing
tick : time step of last state change
id : a numeric ID
ever : a value indicating number of times agent has fled
d : decision function value at current time
c : 1 or 0 specifying whether you were part of the perturbation chain
history : left inhibition vector
previous: [x, y] in previous time step

Returns
-------
Object of class Agent.

"""
struct Agent
    x::Float64
    y::Float64
    h::Float64
    s::Int
    tick::Int
    id::Int
    ever::Int
    d::Float64
    c::Int
    history::MVector{20, Float64}
    previous::MVector{2, Float64}
end


"""
Randomly initialize agents' positions in continuous space.

Parameters
----------
n : number of agents in simulation.
landscape : a binary matrix assigned in global scope

Returns
-------
{Float} array of x and y positions.

"""
function placeOnPatch(n::Int, reefSize::Float64, minDist = 0.05, windowSize = 0.5)
    nSeeds = maximum([round(Int, n / 4), 1])
    seeds  = 1:nSeeds
    cand   = [zeros(n) zeros(n)]
    for s in 1:n
        if s ∈ seeds
            cand[s, :] .= rand(Uniform(0.0 + (0.1 * reefSize), reefSize - (0.1 * reefSize)), 2)
        else
            mySeed = cand[sample(seeds, 1)[1], :]
            cand[s, 1]  = mySeed[1] + rand(Uniform(-windowSize, windowSize))
            cand[s, 2]  = mySeed[2] + rand(Uniform(-windowSize, windowSize))
        end
        td       = pairwise(Euclidean(), cand', cand')
        tooclose = getindex.(findall(y->0<y<minDist, triu(td)), [1 2])
        baddies  = unique(tooclose)
        if size(baddies, 1) > 0
            for h in 1:size(baddies, 1)
                curr = baddies[h, 1]
                while any(0.0 .< td[:, curr] .< minDist)
                    mySeed = cand[sample(seeds, 1)[1], :]
                    cand[curr, 1]  = mySeed[1] + rand(Uniform(-windowSize, windowSize))
                    cand[curr, 2]  = mySeed[2] + rand(Uniform(-windowSize, windowSize))
                    td = pairwise(Euclidean(), cand', cand')
                end
            end
        end
    end
    return cand
end


"""
Randomly choose agents to charge initially.

Parameters
----------
n : number of agents in simulation.
chargers : number of desired chargers at start of the model.

Returns
-------
{Int} array of length n, where 1 indicates agent is fleeing initially.

"""
function chooseCharger(n::Int, chargers::Int)
    cA = round.(Int, zeros(n))
    cA[sample(1:n, chargers, replace = false)] .= 1
    return cA
end


"""
Randomly choose target for charger.

Parameters
----------
n : number of agents in simulation.
chargers : number of desired chargers at start of the model.

Returns
-------
{Int} array of length n, where 1 indicates agent is fleeing initially.

"""
function chooseTarget(agents::Vector{Agent}, charger::Int, cantCharge = 0.0)
    coords = transpose([(p->p.x).(agents) (p->p.y).(agents)])   ## all agent locations
    point  = coords[:, charger]
    dists  = colwise(Euclidean(), point, coords)
    check  = 1
    minD   = minimum(dists[dists .> 0])
    minN   = findallF(y->y==minD, dists)[1]
    while minD < cantCharge
        check += 1
        minD   = smallestN(dists[dists .> 0], check)
        minN   = findallF(y->y==minD, dists)[1]
    end
    return minN, [coords[1, minN] coords[2, minN]]
end


"""
Compute distance between target and charger

Parameters
----------
n : number of agents in simulation.
chargers : number of desired chargers at start of the model.

Returns
-------
{Int} array of length n, where 1 indicates agent is fleeing initially.

"""
function targetChargerDistance(agents::Vector{Agent}, charger::Int, targetLocation::Array{Float64})
    coords = transpose([(p->p.x).(agents) (p->p.y).(agents)])
    point  = coords[:, charger]
    dists  = colwise(Euclidean(), point, targetLocation')
    minD   = minimum(dists[dists .> 0])
    return minD
end


"""
Initialize the agent locations and states.

Parameters
----------
n : number of agents in simulation.
fleeing : number of chargers defined for start of the model.
duration : number of time steps that a flee lasts.
xdim : size in x dim of landscape, assigned in global scope
ydim : size in y dim of landscape, assigned in global scope

Returns
-------
{Agent} vector with initial conditions defined.

"""
function init(n::Int, duration::Int, reefSize::Float64)
    p₀ = placeOnPatch(n, reefSize)
    agents = [Agent(
            p₀[i, 1],                         ## location of an agent
            p₀[i, 2],                         ## location
            Float64.(rand(Uniform(-π, π))),   ## heading
            0,                                ## state
            -duration,                        ## tick
            i, 0, -10.0,                      ## id, ever, initial d
            0,                                ## Boolean whether you are part of cascade
            zeros(20),                        ## history vector
            zeros(2)) for i in 1:n]           ## Dummy [x, y]       
    return agents
end


"""
[2] Updating states and positions __________________________________________________________________________________________________________________________________________

"""


"""
Check to kill model if everything's calmed down.

Parameters
----------
states : {Int} array of current agnet states.
changes : {Int} array of current agent tick field.
ticker : current time step.
duration : number of time steps that a flee lasts.

Returns
-------
{Int} that indicates whether the model should terminate.

"""
function killCheck(states::Array{Int64}, changes::Array{Int64}, ticker::Int, chargeTime::Int, duration = 23)
    k = ifelse(any(states .== 1) || ticker < (maximum(changes) + 5) || ticker < (chargeTime + duration), 0, 1)
    return k
end


"""
Move one agent for one time step.

Parameters
----------
a : {Agent} object.
w₀ : Step size for feed movement.
w₁ : Step size for flee movement.
u₀ : Turning parameter for feed movement. β parameter of Laplace distribution.
u₁ : Turning parameter for flee movement. σ of Normal distribution.

Returns
-------
{Agent} object with updated position and heading.

"""
move(a::Agent, w₀::Float64, w₁::Float64, u₀::Float64, u₁::Float64, b::Float64) =
if a.s == 0
    Φ = a.h + rand(Normal(0, u₀))
    Agent(a.x + rand(Uniform(0, w₀))*cos(Φ), a.y + rand(Uniform(0, w₀))*sin(Φ), Φ, a.s, a.tick, a.id, a.ever, a.d, a.c, a.history, [a.x, a.y])
else
    Φ = a.h + rand(Normal(-b/180, u₁))
    Agent(a.x + w₁*cos(Φ), a.y + w₁*sin(Φ), Φ, a.s, a.tick, a.id, a.ever, a.d, a.c, a.history, [a.x, a.y])
end


"""
Move one charger for one time step.

Parameters
----------
a : {Agent} object.
w₀ : Step size for feed movement.
w₁ : Step size for flee movement.
u₀ : Turning parameter for feed movement. β parameter of Laplace distribution.
u₁ : Turning parameter for flee movement. σ of Normal distribution.

Returns
-------
{Agent} object with updated position and heading.

"""
moveCharger(a::Agent, w₁::Float64, targetLocation::Array{Float64}) =
if a isa Agent
    hx = a.x + cos(a.h)                       
    hy = a.y + sin(a.h)
    headerDiff = atan(hy - a.y, hx - a.x)
    preAngle   = atan(targetLocation[2] - a.y, targetLocation[1] - a.x) - headerDiff
    angle      = atan(sin(preAngle), cos(preAngle))
    Φ = a.h + (angle / 2)
    Agent(a.x + w₁*cos(Φ), a.y + w₁*sin(Φ), Φ, a.s, a.tick, a.id, a.ever, a.d, a.c, a.history, [a.x, a.y])
end


"""
Apply move to all agents for one time step.

Parameters
----------
agents : {Agent} vector.
w₀ : Step size for feed movement.
w₁ : Step size for flee movement.
u₀ : Turning parameter for feed movement. β parameter of Laplace distribution.
u₁ : Turning parameter for flee movement. σ of Normal distribution.

Returns
-------
{Agent} vector with updated positions and headings.

"""
function moveAll!(agents::Vector{Agent}, w₀, w₁, u₀, u₁, b, charger::Int, targetLocation::Array{Float64}, tcDistance::Float64, collision::Int, n::Int, collisionDistance = 0.05, chargerSpeed = 0.03)
    @inbounds begin
        Threads.@threads for (i, agent) in collect(enumerate(agents))
            if i == charger && i ≤ n && tcDistance > collisionDistance && collision == 0 && agents[i].s == 1
                agents[i] = moveCharger(agent, chargerSpeed, targetLocation)
            elseif i ≤ n
                agents[i] = move(agent, w₀, w₁, u₀, u₁, b[i])
            end
        end
    end
end


"""
All possible state changes.

Parameters
----------
a : {Agent} object.
time : current time step.
newInfo : current decision function value.

Returns
-------
{Agent} object with updated state and information.

"""
charge(a::Agent, time::Int)     = Agent(a.x, a.y, a.h, 1, time, a.id, a.ever, a.d, 1, a.history, a.previous)
discharge(a::Agent, time::Int)  = Agent(a.x, a.y, a.h, 0, time, a.id, a.ever, a.d, a.c, a.history, a.previous)
infect(a::Agent, time::Int, dom::Int) = Agent(a.x, a.y, a.h, 1, time, a.id, a.ever + 1, a.d, dom, a.history, a.previous)
recover(a::Agent, time::Int)    = Agent(a.x, a.y, a.h, 0, time, a.id, a.ever, a.d, a.c, a.history, a.previous)
inform(a::Agent, info::Float64) = Agent(a.x, a.y, a.h, a.s, a.tick, a.id, a.ever, info, a.c, a.history, a.previous)
inhibit(a::Agent, newHistory::MVector{20, Float64}) = Agent(a.x, a.y, a.h, a.s, a.tick, a.id, a.ever, a.d, a.c, newHistory, a.previous)


"""
Apply left/right inhibition rolling average values for all agents.

Parameters
----------
agents : {Agent} vector.
time : current time step.
duration : number of time steps that a flee lasts.
decision : decision function values for each agent.

Returns
-------
{Agent} object with updated state and information.

"""
function inhibitAll!(agents::Vector{Agent}, newHistories::Array{Float64}, memoryLength = 20)
    @inbounds begin
        Threads.@threads for (i, agent) in collect(enumerate(agents))
            memOld   = copy(agent.history)
            memShift = copy(memOld[2:memoryLength])
            memOld[1:(memoryLength - 1)] .= memShift
            memOld[memoryLength] = newHistories[i]
            agents[i] = inhibit(agent, memOld)
        end
    end
end


"""
Apply state update to one agent.

Parameters
----------
a : {Agent} object.
time : current time step.
duration : number of time steps that a flee lasts.
cooldown : number of time steps needed before agent can be re-excited.

Returns
-------
{Agent} object with updated state and information.

"""
update(a::Agent, time::Int, duration::Int, dom::Int) =
if a.s == 0
    roll = rand(Bernoulli(responseProb(a.d)))
    ifelse((time - a.tick) < duration, a, ifelse(roll, infect(a, time, dom), a))
else
    ifelse((time - a.tick) > duration, recover(a, time), a)
end


"""
Apply state update to all agents.

Parameters
----------
agents : {Agent} vector.
time : current time step.
duration : number of time steps that a flee lasts.
decision : decision function values for each agent.

Returns
-------
{Agent} object with updated state and information.

"""
function updateAll!(agents::Vector{Agent}, time::Int, duration::Int, decis::Array{Float64}, dom::Array{Int64})
    @inbounds begin
        Threads.@threads for (i, agent) in collect(enumerate(agents))
            if dom[i] != 0
                χ = dom[i]
            else
                χ = 0
            end
            agents[i] = update(inform(agent, decis[i]), time, duration, χ)
        end
    end
end


"""
[3] Sensing and decision making __________________________________________________________________________________________________________________________________________

"""


"""
Calculate distance and angles for agents within sensory radius.

Parameters
----------
agents : {Agent} vector.
sensoryRange : sensory radius.
n : number of agents.
tempStates : array of current agent states.

Returns
-------
{Float} arrays with the relevant information.

"""
function neighborDist(agents::Vector{Agent}, sensoryRange::Float64, n::Int, tempStates::Array{Int64}, minDist = 0.05)
    Δ  = zeros(n, n)
    η₁ = zeros(n, n)
    η₂ = zeros(n, n)
    coords     = transpose([(p->p.x).(agents) (p->p.y).(agents)])
    coordsPre  = transpose([(p->p.previous[1]).(agents) (p->p.previous[2]).(agents)])
    headings   = (p->p.h).(agents)
    searchTree = KDTree(coords)
    @inbounds begin
        Threads.@threads for i in 1:n
            # if tempStates[i] == 0
            ## Distances
            point = coords[:, i]
            idxs = inrange(searchTree, point, sensoryRange, false)
            deleteat!(idxs, idxs .== i)
            Δ[idxs, i] = colwise(Euclidean(), point, coords[:, idxs])
            ## Current angle
            Bx = point[1] + cos(headings[i])                       
            By = point[2] + sin(headings[i])
            headerDiff    = @. atan(By - point[2], Bx - point[1])
            angles        = @. atan(coords[2, idxs] - point[2], coords[1, idxs] - point[1]) - headerDiff
            @. angles     = atan(sin(angles), cos(angles))
            η₁[idxs, i]   = angles
            ## Previous angle
            anglesPre     = @. atan(coordsPre[2, idxs] - point[2], coordsPre[1, idxs] - point[1]) - headerDiff
            @. anglesPre  = atan(sin(anglesPre), cos(anglesPre))
            η₂[idxs, i]   = anglesPre
            # end
        end
    end
    ## Threshold minimum distance
    @. Δ[(Δ > 0) & (Δ < minDist)] = minDist
    return Δ, η₁, η₂
end


"""
Compute height and width for S matrix

Parameters
----------
Δ : {Float} array of neighbor distances.
μ : agent body size in m.

Returns
-------
{Float} arrays with the relevant information.

"""
function computeHW(Δ::Array{Float64}, μ = 0.2, heightFactor = 0.2)
    H = heightFactor .* μ ./ (2 .* Δ)
    W = μ ./ (2 .* Δ)
    drop = isinf.(W)
    H[drop] .= 0
    W[drop] .= 0
    H .= 2 .* atan.(H) .* (180 / π)
    W .= 2 .* atan.(W) .* (180 / π)
    return H, W
end


"""
Calculate loom matrices from neighbor positions and angles.

Parameters
----------
S₁ : {Float} array of neighbor sizes at time t−1.
S₂ : {Float} array of neighbor sizes at time t−2.
η₁ : {Float} array of angles from focal current to neighbor current.
η₂ : {Float} array of angles from focal current to neighor previous.
d₀ : length of timestep in seconds.

Returns
-------
{Float} arrays with the relevant information.

"""
function computeLT(S₁::Array{Float64}, S₂::Array{Float64}, η₁::Array{Float64}, η₂::Array{Float64}, d₀::Float64)
    @inbounds begin
        ## Loom matrix
        L₁ = (S₁ .- S₂)
        @. L₁ = (abs(L₁) + L₁) / 2
        entries = findall(y->y==0, S₂)
        exits = findall(y->y==0, S₁)
        @. L₁[entries] = 0
        @. L₁[exits] = 0
        @. L₁ = L₁ / d₀
        ## Translation matrix
        ηDiff = (η₁ .- η₂) 
        T₁ = abs.(atan.(sin.(ηDiff), cos.(ηDiff))) .* (180 / π)
        @. T₁[entries] = 0
        @. T₁[exits] = 0
        @. T₁ = T₁ / d₀
    end
    return L₁, T₁
end


"""
Apply occlusion to distance and angle matrices. Sorry that I wrote this
horrible function into existence. I promise to make it neater some day.

Parameters
----------
S₁ : {Float} array of neighbor sizes at time t−1.
L₁ : {Float} array of neighbor looms at time t−1.
T₁ : {Float} array of neighbor translation at time t−1.
Δ₁ : {Float} array of neighbor distance at time t−1.
η₁ : {Float} array of neighbor angles at time t−1.
n : number of agents.

Returns
-------
{Float} arrays with the relevant information.

"""
function applyOcclusion(S::Array{Float64}, Δ::Array{Float64}, η::Array{Float64}, n::Int)
    @inbounds begin
        Threads.@threads for k in 1:n
            ds = Δ[:, k]
            as = η[:, k]
            ss = S[:, k]
            ## Prepare by finding who's ahead and who's behind
            ord = sortperm(ds)
            ord = ord[ds[ord] .!= 0]
            nord = length(ord)
            span = ss[ord] ./ 2
            bounds = [((180 .+ as[ord] .* (180 / π)) .- span) ((180 .+ as[ord] .* (180 / π)) .+ span)]
            bounds[bounds .< 0] .= 360 .- abs.(bounds[bounds .< 0])
            bounds[bounds .> 360] .= bounds[bounds .> 360] .- 360
            rays = copy(bounds)
            for n in 1:(nord - 1)
                ## Get vector of who's behind
                isbehind = rays[:, 1] .> rays[:, 2]
                isbehind = isbehind .* cumsum(isbehind)
                ## If focal is behind
                if isbehind[n] > 0
                    a₁ = rays[n, 1]
                    a₂ = rays[n, 2]
                    for z in (n + 1):nord
                        b₁ = rays[z, 1]
                        b₂ = rays[z, 2]
                        ## If next one is behind
                        if isbehind[z] > 0
                            if (a₁ ≤ b₁ ≤ 360) && (0 ≤ b₂ ≤ a₂)
                                rays[z, 1] = 0
                                rays[z, 2] = 0
                            elseif (b₁ ≤ a₁ ≤ 360) && (0 ≤ b₂ ≤ a₂)
                                rays[z, 2] = a₁
                            elseif (a₁ ≤ b₁ ≤ 360) && (0 ≤ a₂ ≤ b₂)
                                rays[z, 1] = a₂
                            end
                        ## If next one is not behind
                        elseif isbehind[z] == 0
                            if (0 ≤ b₁ ≤ b₂ ≤ a₂) || (a₁ ≤ b₁ ≤ b₂ ≤ 360)
                                rays[z, 1] = 0
                                rays[z, 2] = 0
                            elseif (b₁ ≤ a₁ ≤ b₂)
                                rays[z, 2] = a₁
                            elseif (b₁ ≤ a₂ ≤ b₂)
                                rays[z, 1] = a₂
                            end
                        end
                    end
                ## If focal is not behind
                elseif isbehind[n] == 0
                    a₁ = rays[n, 1]
                    a₂ = rays[n, 2]
                    for z in (n + 1):nord
                        ## If next one is behind
                        if isbehind[z] > 0
                            b₁ = rays[z, 1]
                            b₂ = rays[z, 2]
                            if (a₁ ≤ b₁ ≤ a₂)
                                rays[z, 1] = a₂
                            elseif (a₁ ≤ b₂ ≤ a₂)
                                rays[z, 2] = a₁
                            end
                        ## If next one is not behind
                        elseif isbehind[z] == 0
                            b₁ = rays[z, 1]
                            b₂ = rays[z, 2]
                            if (a₁ ≤ b₁ ≤ b₂ ≤ a₂)
                                rays[z, 1] = 0
                                rays[z, 2] = 0
                            elseif (a₁ ≤ b₁ ≤ a₂)
                                rays[z, 1] = a₂
                            elseif (a₁ ≤ b₂ ≤ a₂)
                                rays[z, 2] = a₁
                            end
                        end
                    end
                end
            end
            ## Assume neighbors with only a small visible sliver are not discernable. This
            ## stabilizes the artefact of tiny movements creating massive looms.
            orderedScalars = mod.((rays[:, 2] .- rays[:, 1]), 360.0) ./ mod.((bounds[:, 2] .- bounds[:, 1]), 360.0)
            orderedScalars[orderedScalars .< 0.10] .= 0.0
            S[ord, k] = S[ord, k] .* orderedScalars
        end
    end
    return S
end


"""
Applies exponential decay filter to inhibition vector.

Parameters
----------
v : {MVector} inhibition vector.
τ : {Float} timescale parameter

Returns
-------
{Float} array with the relevant information.

"""
function decay(v, τ = 0.10, normalization = 3.848732, l = 20, d₀ = 0.03)
    ## Note: norm factor = sum(exp(-x/τ)), where x = 0.03 * 0:19
    td = d₀ .* collect(range(0, stop = l - 1))
    sc = exp.(-td ./ τ) ./ normalization
    wt = v .* reverse(sc)
    return sum(wt)
end


"""
Calculate the decision function value from sensory information.

Parameters
----------
S₁ : {Float} array of neighbor sizes at time t−1.
L₁ : {Float} array of neighbor looms at time t−1.
T₁ : {Float} array of neighbor translation at time t−1.
η₁ : {Float} array of neighbor angles at time t−1.
n : number of agents.
inhibition : scalar that turns down κ₃ and κ₄ (from 0 to 1).
tempInterest : array indictating whether an agent has a flee-er in range.

Note: Empirical SDs are 40 for looming, and 498 for translation.

Returns
-------
{Float} array with the relevant information.

"""
function decisions(agents::Vector{Agent}, L₁::Array{Float64}, T₁::Array{Float64}, η₁::Array{Float64}, n::Int, tempStates::Array{Int64}, inhibition::Int, excitation::Float64, intercept::Float64, transExciteFactor = 0.2162069, loomSD = 40, transSD = 498)
    d     = fill(zero(Float64), n)
    m     = fill(zero(Int64), n)
    chain = (p->p.c).(agents)
    @inbounds begin
        Threads.@threads for i in 1:n
            if tempStates[i] == 0
                ## First sum each neighbor's motion for excitation
                ## Loom SD: 40, Trans SD: 498
                exciteAll = (L₁[:, i] ./ loomSD) .+ (transExciteFactor .* T₁[:, i] ./ transSD)
                excite    = sum(exciteAll)
                whoIsSeen = findallF(y->y!=0, η₁[:, i])
                ## Calculate decision values
                if inhibition == 0
                    dec   = excitation * excite
                elseif inhibition == 1
                    inhib = decay(agents[i].history)
                    dec   = excitation * excite / (0.022 + inhib)
                end
                ## Decision value
                d[i]      = intercept + dec
                m[i]      = sum(chain[whoIsSeen]) > 0
            end
        end
    end
    return d, m
end


"""
Convert decision scores into startle probabilities.

Parameters
----------
ρ : decision values for agents

Returns
-------
P : probabilities for Bernoulli

"""
function responseProb(ρ::Float64)
    P = 1 / (1 + exp(-ρ))
    return P
end


"""
[4] Simulation and output __________________________________________________________________________________________________________________________________________

"""


"""
Output full data for making movies.

Parameters
----------
agents : {Agent} vector.
n : number of agents.
maxtime : maximum number of time steps allowed.

Returns
-------
{Float} array with the relevant information.

"""
function extractFullData(agents::Vector{Agent}, n::Int, maxtime::Int, charger::Int, target::Int)
    @inbounds begin
        out = fill(zero(Float64), maxtime * n, 10)
        nₒ = size(agents, 1) - n
        Threads.@threads for i in 1:nₒ
            a = agents[i]
            out[i, 1]  = a.x::Float64
            out[i, 2]  = a.y::Float64
            out[i, 3]  = a.h::Float64
            out[i, 4]  = a.s::Int
            out[i, 5]  = a.id::Int
            out[i, 6]  = a.ever::Int
            out[i, 7]  = a.d::Float64
            out[i, 8]  = a.c::Int
            out[i, 9]  = ifelse(a.id == charger, 1, 0)
            out[i, 10] = ifelse(a.id == target, 1, 0)
        end
    end
    return out
end


"""
Main workhorse that performs one simulation.

Parameters
----------
All parameters are described above.

Returns
-------
{Float} array with the relevant information.

"""
function scaredyFish(
        n::Int, 
        fleeing::Int, 
        w₀::Float64, 
        w₁::Float64, 
        u₀::Float64,
        u₁::Float64,
        sensoryRange::Float64,
        duration::Int, 
        maxtime::Int,
        d₀::Float64,
        inhibition::Int,
        excitation::Float64,
        intercept:: Float64,
        reefSize::Float64,
        transInhibFactor = 0.2162069,
        loomSD = 40, 
        transSD = 498)
    ## Initiate some vectors and tickers
    ticker         = 0
    killer         = 0
    collision      = 0
    charger        = findallF(y->y==1, chooseCharger(n, fleeing))[1]
    chargeTime     = 21
    target         = 0
    tcDistance     = 1e6
    targetLocation = [0.0 0.0]
    ## Burn in: a small # of time steps to get some initial motion and sensory values
    agents     = init(n, duration, reefSize)
    b          = zeros(n)
    Δ₂, F0, F1 = neighborDist(agents, sensoryRange, n, (p->p.s).(agents))
    H₂, W₂     = computeHW(Δ₂)
    W₂         = applyOcclusion(W₂, Δ₂, F0, n)
    S₂         = sqrt.(H₂ .* W₂)
    moveAll!(agents, w₀, w₁, u₀, u₁, b, n + 1, [0.0 0.0], tcDistance, collision, n)
    Δ₁, η₁, η₂ = neighborDist(agents, sensoryRange, n, (p->p.s).(agents))
    H₁, W₁     = computeHW(Δ₁)
    W₁         = applyOcclusion(W₁, Δ₁, η₁, n)
    S₁         = sqrt.(H₁ .* W₁)
    L₁, T₁     = computeLT(S₁, S₂, η₁, η₂, d₀)
    ## Go ahead and smash that record button
    res = copy(agents)
    ## Workhorse: start simulation loop
    while ticker < maxtime && killer == 0
        ## Update agents position, state, and sensory info
        ticker    += 1
        tempStates = (p->p.s).(agents)
        tempTicks  = (p->p.tick).(agents)
        ## Update sensory information
        inhibitAll!(agents, (sum(L₁, dims = 1) ./ loomSD) .+ (transInhibFactor .* sum(T₁, dims = 1) ./ transSD))
        ## Kick off the cascade with a false alarm
        if ticker == chargeTime
            agents[charger] = charge(agents[charger], ticker)
            target, targetLocation = chooseTarget(agents, charger)
        end
        ## Update charger distance if charging
        if ticker ≥ chargeTime
            tcDistance     = targetChargerDistance(agents, charger, targetLocation)
            targetLocation = [agents[target].previous[1] agents[target].previous[2]]
            ρ, δ           = decisions(agents, L₁, T₁, η₁, n, tempStates, inhibition, excitation, intercept)
            b              = sum(S₁ .* (η₁ .> 0), dims = 1) .- sum(S₁ .* (η₁ .≤ 0), dims = 1)
            updateAll!(agents, ticker, duration, ρ, δ)
        end
        ## Terminate the charge if there is a collision
        if tcDistance ≤ 0.05 && collision == 0
            agents[charger] = discharge(agents[charger], ticker)
            collision       = 1
        end
        ## Update positions and prepare sensory for next time step
        moveAll!(agents, w₀, w₁, u₀, u₁, b, charger, targetLocation, tcDistance, collision, n)
        S₂         = copy(S₁)
        Δ₁, η₁, η₂ = neighborDist(agents, sensoryRange, n, (p->p.s).(agents))
        H₁, W₁     = computeHW(Δ₁)
        W₁         = applyOcclusion(W₁, Δ₁, η₁, n)
        S₁         = sqrt.(H₁ .* W₁)
        L₁, T₁     = computeLT(S₁, S₂, η₁, η₂, d₀)
        res        = vcat(res, agents)
        killer     = killCheck(tempStates, tempTicks, ticker, chargeTime)
    end
    ## Output results
    out = extractFullData(res, n, ticker, charger, target)
    return out
end;




