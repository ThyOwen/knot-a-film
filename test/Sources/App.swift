import SwiftUI
import CoreML


@main struct TestApp : App {
    @State private var startDate = Date.now
    var body: some Scene {
        WindowGroup {
            //MetalView()
            //NBodyView()
            Circle()
                .onAppear {
                    MetalTests.sumTest()
                }
        }
    }
}

/*
@main struct TestApp {
    static func main() {
        var graph = GraphNew(numNodes: 4)
        graph.buildGraphStandard()
    }
}
*/
