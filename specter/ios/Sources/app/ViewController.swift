import UIKit
import SwiftUI
import SnapKit
import ios_Base

class ViewController: UIViewController {
    
    let moduleService: ModuleService
    
    override init(nibName nibNameOrNil: String?, bundle nibBundleOrNil: Bundle?) {
        weak var weakSelf: ViewController?
        let defaultCameraOutputActions = CameraStreamViewController.OutputActions(
            didReceiveNewBuffer: {}
        )
        self.moduleService = ModuleService(modules: allModules) { module in
            let cameraVC = CameraStreamViewController(outputActions: defaultCameraOutputActions)
            let moduleVC = modify(CommonModuleViewController(cameraModule: cameraVC)) {
                $0.modalPresentationStyle = .overFullScreen
            }
            weakSelf?.present(moduleVC, animated: true)
        }

        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
        weakSelf = self
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .red
        
        let hostingController = UIHostingController(
            rootView: MainView(moduleService: moduleService)
        )
        addChild(hostingController)
        view.addSubview(hostingController.view)
        hostingController.view.snp.makeConstraints { (make) -> Void in
            make.leading.trailing.equalTo(self.view)
            make.top.bottom.equalTo(self.view)
         }
    }
}
