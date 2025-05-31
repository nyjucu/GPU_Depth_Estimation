import torch 
import tokkyutils as tu
import tokyymodel as tm
import tokyyloss as tl
import tokyydataset as tds
from torchsummary import summary 
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    print(torch.__version__)

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

    print( f"Using device: { device }" )

    dev_number = torch.cuda.current_device()

    print( f"Current device number is: { dev_number }" )

    dev_name = torch.cuda.get_device_name()

    print( f"Current device name is: { dev_name }" )

    encoder = tm.Encoder().to( device )
    resunet = tm.ResUNet().to( device )
    dummy_input = torch.randn( 1, 3, 128, 128 ).to( device )

    summary( resunet, input_size = ( 3, 128, 128 ), batch_size = 2 )

    ########
    # LOAD #
    ########

    accum_steps = 2
    batch_size = 96
    num_epochs = 20


    # load_dataset = tu.ask_yes_no("Load dataset?")

    # if load_dataset:
    print("Loading dataset...")
    train_dataset = tds.DepthDataset("nyu/train", transform=tds.rgb_transform, depth_transform=tds.depth_transform)
    val_dataset = tds.DepthDataset("nyu/val", transform=tds.rgb_transform, depth_transform=tds.depth_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("Dataset loaded succesfully!")

    #########
    # TRAIN #
    #########

    # train_model = tu.ask_yes_no("Train model?")

    # if train_model: 
    print("Training model...")
    model = tm.ResUNet().to( device )
    criterion = tl.AWLoss( lambda_depth = 0.5 ).to( device )
    optimizer = torch.optim.Adam( model.parameters(), lr = 1e-4 )

    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0


        loop = tqdm( enumerate( train_loader ), total = len( train_loader ), desc = f"Epoch { epoch + 1 } / { num_epochs }" )

        for i, ( rgb, depth ) in loop:
            rgb = rgb.to( device, non_blocking=True )
            depth = depth.to( device, non_blocking=True )

            with autocast( device_type=device.type ):
                outputs = model( rgb )
                loss = criterion( outputs , depth ) / accum_steps

            scaler.scale(loss).backward()

            if ( i + 1 ) % accum_steps == 0 or ( i + 1 ) == len( train_loader ):
                scaler.step( optimizer )
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * rgb.size( 0 )

            loop.set_postfix(loss=loss.item())

        train_loss /= len( train_loader.dataset )

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb = rgb.to( device, non_blocking=True )
                depth = depth.to( device, non_blocking=True )
                
                with autocast( device_type=device.type ):
                    outputs = model( rgb )
                    loss = criterion( outputs, depth )

                val_loss += loss.item() * rgb.size( 0 )

        val_loss /= len( val_loader.dataset )

        print( f"Epoch [ { epoch + 1 } / { num_epochs } ], Train Loss: { train_loss: .4f }, Val Loss: { val_loss: .4f }" )

    print("Model trained succesfully!")

    ########
    # SAVE #
    ########

    # save_model = tu.ask_yes_no("Save model?")

    # if save_model:
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved succesfully!")

if __name__ == '__main__':
    main()
    