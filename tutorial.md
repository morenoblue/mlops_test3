# Steps to connect to the ssh into the DEI machine using tailscale

## Step 1 - Create an account on tailscale.com

1. Go to tailscale.com
2. Create an account there

## Step 2 - Generate a authkey

1. In the tailscale admin panel home, click on the option that says `settings`
2. Now while on the `settings` click on the option that says `keys`. This will 
   open a ui where you can create an `authkey`, pick the definitions you want for 
   the key and then store it somewhere where you can reference it later.

## Step 3 - Install tailscale on your personal machine

note: I'm (nuno) using linux so for windows the installation might be different, research it yourself if you need to.

1. Open your terminal an run this:

```sh
curl -fsSL https://tailscale.com/install.sh | sh
```
2. Then enable the service:

```sh
sudo systemctl enable --now tailscaled
```

3. Now run it:

```sh
sudo tailscale up
```

4. Check your ip provided by tailscale:

```sh
tailscale ip -4
```

## Step 4 - Install tailscale on the VM

1. Open your terminal an run this:

```sh
curl -fsSL https://tailscale.com/install.sh | sh
```
2. Then enable the service:

```sh
sudo systemctl enable --now tailscaled
```

3. Remember the `authkey` that you stored? yes, now its the time for you to use 
   it. Copy it into the command below and run that:

```sh
sudo tailscale up --authkey=`authkey`
```

4. You can now also check the VM ip provided by tailscale:

```sh
tailscale ip -4
```

## Step 5 - Make sure that the ssh is configured in the VM

1. Run this to check if ssh is running:

```sh
systemctl status ssh
```

2. If you read in the output that its not running then do this:

```sh
systemctl start ssh
```

## Step 6 - Connect to the VM through ssh via the Tailscale IP

1. From your personal computer you just need to run the command below. Remember 
   that to obtain the `ip_from_vm_provided_by_tailscale` you just need to run 
   `tailscale ip -4` on the VM.

```sh
ssh admin@`ip_from_vm_provided_by_tailscale`
```

After that add the password and done! you are now connected to the vm via your own machine.




# Steps to free the memory you have asked for on the VM

For reference check in the beginning the outputs of these commands:

```sh
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
sudo vgs
sudo lvs -a -o +devices
```

Notice how `xvdb` is free. meaning that space isn't being used. We need to turn 
it into a LVM physical volume:

```sh
sudo pvcreate /dev/xvdb
```

Now to confirm that xvdb appears as a PV, compare the outputs of the command 
below with the initial output of it:

```sh
sudo pvs
```

Now you should add the PV to your volume group:

```sh
sudo vgextend ubuntu-vg /dev/xvdb
```

Using the command below verify that VSIZE has jumped in size (in my case from ~23G to ~48G)

```sh
sudo vgs
```

Now just extend the root logical volume:

```sh
sudo lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
```

At this point your root must already be using your full available space. To 
verify you can run these commands and check their output:

```sh
df -h /
sudo lvs -o lv_name,vg_name,lv_size,devices
sudo vgs
```
