# Setup NewRiver

[General Information](https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples)
[Group Web](https://mlp.ece.vt.edu/wiki/doku.php)

## Setting password-less SSH
From personal computer:
```
scp -r <userid>@godel.ece.vt.edu:/srv/share/lab_helpful_files/ ~/
```
Change ``<userid>`` to your CVL account username in the ~/lab_helpful_files/config file and move it to ~/.ssh
```
mv ~/lab_helpful_files/config ~/.ssh/
ssh-keygen -t rsa
```
Enter the password here. Make sure ~/ on sever has .ssh folder: login, does

```
cd ~/.ssh
```
work? if not, type
```
mkdir .ssh
scp ~/.ssh/id_rsa.pub godel:~/.ssh/
```
On server:

```
cd ~/.ssh/
cat id_rsa.pub >> authorized_keys2
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys2
```
Now you should be able to type something like ``$ ssh huck`` On your personal computer and it will login without asking for a password.









## Preparation

- Install miniconda
```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
```

- Create an environment named TF (or whatever name you want)
```
source ~/.bashrc
conda create -n TF
```

- Install Caffe
