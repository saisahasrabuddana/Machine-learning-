import statistics
data = list(map(int,input("Enter numbers: ").split()))

mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)
variance = statistics.variance(data)
st_deviation = statistics.stdev(data)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Varience:", variance)
print("Standard Deviation:",st_deviation)