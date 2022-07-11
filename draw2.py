import dill 
import matplotlib.pyplot as plt








if __name__ == '__main__':
    with open("Compression, l2_norm=1.3noise_multiplier=0.01epochs=1num_microbatch200local_updates=5shuffle_rate=0.2percentile=0.05", "rb") as dill_file:
        a = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.05epochs=1num_microbatch200local_updates=5shuffle_rate=0.2percentile=0.05", "rb") as dill_file:
        b = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.1epochs=1num_microbatch200local_updates=5shuffle_rate=0.2percentile=0.05", "rb") as dill_file:
        c = dill.load(dill_file)
    


    with open("Compression, l2_norm=1.3noise_multiplier=0.01epochs=1num_microbatch200local_updates=5shuffle_rate=0.5percentile=0.05", "rb") as dill_file:
        d = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.05epochs=1num_microbatch200local_updates=5shuffle_rate=0.5percentile=0.05", "rb") as dill_file:
        e = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.1epochs=1num_microbatch200local_updates=5shuffle_rate=0.5percentile=0.05", "rb") as dill_file:
        f = dill.load(dill_file)


    with open("Compression, l2_norm=1.3noise_multiplier=0.01epochs=1num_microbatch200local_updates=5shuffle_rate=0.8percentile=0.05", "rb") as dill_file:
        h = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.05epochs=1num_microbatch200local_updates=5shuffle_rate=0.8percentile=0.05", "rb") as dill_file:
        i = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.1epochs=1num_microbatch200local_updates=5shuffle_rate=0.8percentile=0.05", "rb") as dill_file:
        j = dill.load(dill_file)


    with open("Compression, l2_norm=1.3noise_multiplier=0.01epochs=1num_microbatch200local_updates=5shuffle_rate=0.9percentile=0.05", "rb") as dill_file:
        k = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.05epochs=1num_microbatch200local_updates=5shuffle_rate=0.9percentile=0.05", "rb") as dill_file:
        l = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.1epochs=1num_microbatch200local_updates=5shuffle_rate=0.9percentile=0.05", "rb") as dill_file:
        m = dill.load(dill_file)


    with open("Compression, l2_norm=1.3noise_multiplier=0.01epochs=1num_microbatch200local_updates=5shuffle_rate=1percentile=0.05", "rb") as dill_file:
        n = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.05epochs=1num_microbatch200local_updates=5shuffle_rate=1percentile=0.05", "rb") as dill_file:
        o = dill.load(dill_file)
    with open("Compression, l2_norm=1.3noise_multiplier=0.1epochs=1num_microbatch200local_updates=5shuffle_rate=1percentile=0.05", "rb") as dill_file:
        p = dill.load(dill_file)

    fig = plt.figure()

    sub1 = fig.add_subplot(2, 3, 1, )
    sub2 = fig.add_subplot(2, 3, 2, sharey=sub1)
    sub3 = fig.add_subplot(2, 3, 3, sharey=sub1)
    sub4 = fig.add_subplot(2, 3, 4, sharey=sub1)
    sub5 = fig.add_subplot(2, 3, 5, sharey=sub1)

    sub1.plot(a, color="cyan")
    sub1.plot(b, color="red")
    sub1.plot(c, color="pink")
    sub1.title.set_text('80% Similarity')
    
    sub2.plot(d, color="cyan")
    sub2.plot(e, color="red")
    sub2.plot(f, color="pink")
    sub2.title.set_text('50% Similarity')

    sub3.plot(h, color="cyan")
    sub3.plot(i, color="red")
    sub3.plot(j, color="pink")
    sub3.title.set_text('20% Similarity')

    sub4.plot(k, color="cyan")
    sub4.plot(l, color="red")
    sub4.plot(m, color="pink")
    sub4.title.set_text('10% Similarity')

    sub5.plot(n, color="cyan")
    sub5.plot(o, color="red")  
    sub5.plot(p, color="pink")  
    sub5.title.set_text('0% Similarity')  
    
    plt.plot()
    # plt.ylabel('training loss')
    plt.ylabel('testing accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()