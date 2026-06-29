#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/utility/Setting.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/BuiltinOps.h"

namespace utility = lingodb::utility;
namespace {
using namespace lingodb::compiler::dialect;

// When true, wraps eligible subop.materialize ops in subop.mini_buffer before split-into-steps.
// Default off until workloads are routinely validated with minibuffer enabled.
utility::GlobalSetting<bool> miniBufferEnabled("system.subop.mini_buffer", false);

class InsertMiniBufferPass : public mlir::PassWrapper<InsertMiniBufferPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMiniBufferPass)
   llvm::StringRef getArgument() const override { return "subop-insert-mini-buffer"; }

   void runOnOperation() override {
      if (!miniBufferEnabled.getValue()) {
         return;
      }
      llvm::SmallVector<subop::MaterializeOp> materializes;
      getOperation()->walk([&](subop::MaterializeOp mat) {
         materializes.push_back(mat);
         return mlir::WalkResult::advance();
      });
      llvm::sort(materializes, [](subop::MaterializeOp a, subop::MaterializeOp b) { return a->isBeforeInBlock(b); });

      for (subop::MaterializeOp mat : llvm::reverse(materializes)) {
         auto group = mat->getParentOfType<subop::ExecutionGroupOp>();
         if (!group) {
            continue;
         }
         if (mat->getParentRegion() != &group.getSubOps()) {
            continue;
         }
         if (mat->getParentOfType<subop::MiniBufferOp>()) {
            continue;
         }
         mlir::Value stream = mat.getStream();
         if (!stream.getDefiningOp() || !stream.hasOneUse()) {
            continue;
         }
         if (mlir::isa<subop::MiniBufferOp>(stream.getDefiningOp())) {
            continue;
         }

         mlir::OpBuilder b(mat);
         mlir::Location loc = mat.getLoc();
         // Batch size for GrowingBuffer before flushing downstream. Must match lowering: flush loop clones
         // downstream ops per element (see MiniBufferLowering); tail flush handles the final partial batch.
         constexpr uint64_t kBatch = 16;
         auto miniBuf = b.create<subop::MiniBufferOp>(loc, stream, kBatch);
         mlir::Block* body = b.createBlock(&miniBuf.getRegion(), {}, stream.getType(), mlir::ArrayRef<mlir::Location>{loc});
         mat->moveBefore(body, body->end());
         mat.setOperand(0, body->getArgument(0));
         b.setInsertionPointToEnd(body);
         b.create<tuples::ReturnOp>(loc);
      }
   }
};
} // namespace

namespace lingodb::compiler::dialect::subop {
std::unique_ptr<mlir::Pass> createInsertMiniBufferPass() {
   return std::make_unique<InsertMiniBufferPass>();
}
} // namespace lingodb::compiler::dialect::subop
