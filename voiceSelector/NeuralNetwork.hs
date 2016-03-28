module NeuralNetwork (
  fromList,
  solve,
  empty,
  progresive,
  sigmoid,
  NN,
  solve',
  fun,
  ignorefirst,
  caseBackpropagation,
  backpropagation,
  bp,
  fromRandomList
) where

import qualified Data.Matrix as A
import Data.List (foldl')

sigmoid x = 1 / (1 + exp (-1 * x))

addBias :: (Num a) => A.Matrix a -> A.Matrix a
addBias x = (A.<->) (A.fromList 1 (A.ncols x) (repeat 1)) x

newtype NN a = NN [A.Matrix a]

instance (Show a) => Show (NN a) where
  show (NN a) = show $ map (A.toLists) a

-- transforma una lista de lista de lista de numeros a red neural
fromList list = 
  let temp = (map (map (\x-> (A.fromList) 1 (length x) x)) list) in
  NN (map (foldl1 (A.<->)) temp)

solve' :: (Num a, Floating a) => NN a -> A.Matrix a -> A.Matrix a
solve' (NN net) input = foldl' (help) (addBias input) net
  where help x y = addBias $ fmap (sigmoid) (y*x)

-- Resuelve una red neural con un input dado
solve :: (Num a, Floating a) => NN a -> [a] -> [a]
solve net input = drop 1 $ A.toList $ solve' net (A.fromList (length input) 1 input)

-- lo mismo que solve' pero retorna los resultados de cada capa de neuronas
-- chevere pa analisis
progresive :: (Num a, Floating a) => NN a -> A.Matrix a -> [A.Matrix a]
progresive (NN net) input = scanl (help) (addBias input) net
  where help x y = addBias $ fmap (sigmoid) (y*x)

-- Retorna una red en la que todos los pesos son cero, donde
-- input es un int que dice cuantos datos tiene el input
-- l es una lista, donde cada elemento es la cantidad de neuronas de cada capa, la cantidad de capas de la red depende del tamano de este arreglo
empty input l = NN $ help input l
  where
    help _ [] = []
    help i (l:lx) = (A.zero l (i+1)):(help l lx)

fromRandomList input l rand = NN $ help input l rand
  where
    help _ [] _ = []
    help i (x:xs) r = (A.fromList x (i+1) r):(help x xs (drop (x+i+1) r))

testToMatrix x =
  let help = map (\(x,y) -> (A.fromList 1 (length x) x, A.fromList 1 (length y) y)) x in
  foldl1 (aux) help
  where
    aux (x,y) (z,w) = (x A.<-> z,y A.<-> w)

-- primer argumento: tasa de aprendizaje
-- segundo la red que va a aprender
-- tercero los casos de prueba
-- cuarto una funcion que dada la red que se ha entrenado hasta el momento, diga si se deja de entrenar o no
-- backpropagation :: (Num a, Floating a) => a -> NN a -> [([a],[a])] -> (NN a -> Bool) -> A.Matrix a
-- backpropagation tasa net testCases stopFun
--   | stopFun net = ((netoutput net) - results)
--   | otherwise = ((netoutput net) - results)
--   where
--     parse = testToMatrix testCases
--     results = addBias $ A.transpose $ snd $ parse
--     outputs x = progresive x $ A.transpose $ fst $ parse
--     netoutput x = last $ outputs x

(<**>) :: (Num a) => A.Matrix a -> A.Matrix a -> A.Matrix a
a <**> b = A.elementwise (*) a b

ignorefirst x = A.submatrix 2 (A.nrows x) 1 1 x

fun hid (d, (x:xs), l) = val:l `seq` (A.transpose val,xs,val:l)
  where
    val = hid<**>(fmap ((-) 1) hid)<**>(ignorefirst $ A.transpose (d*x))

caseBackpropagation :: (Num a, Floating a) => a -> NN a -> (A.Matrix a,A.Matrix a) -> NN a
caseBackpropagation tasa net@(NN arr) caso = NN final
  where
    progre = progresive net $ fst $ caso
    outputs = map ignorefirst $ tail progre
    result = snd $ caso
    netoutput = last outputs
    hiddens = init outputs
    dk = netoutput <**> (fmap ((-) 1) netoutput) <**> (result-netoutput)
    d = ((\(_,_,x)->x) (foldr fun ((A.transpose dk),reverse arr,[]) hiddens)) ++ [dk]
    deltaw = map (fmap ((*) tasa)) $ zipWith (*) d (fmap (A.transpose) $ init progre)
    final = zipWith (+) arr deltaw

bp :: (Num a, Floating a) => a -> NN a -> [([a],[a])] -> [NN a]
bp tasa net testCases = cases `seq` backpropagation tasa net cases
  where
    cases = map (\(x,y)->(A.fromList (length x) 1 x, A.fromList (length y) 1 y)) testCases

backpropagation :: (Num a, Floating a) => a -> NN a -> [(A.Matrix a,A.Matrix a)] -> [NN a]
backpropagation tasa net testCases = net `seq` net:(backpropagation tasa myNet testCases)
  where
    myNet = foldl' (caseBackpropagation tasa) net testCases


















